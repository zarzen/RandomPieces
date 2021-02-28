#include "connection.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include "kernels.h"
#include "logger.h"
#include "utils.h"
#include <assert.h>

#define SOCK_MIN_SIZE (64*1024)

ConnectionType_t NetConnection::getType() {
  return Net;
}

void NetConnection::initBuffer() {
  close = false;
  cudaSetDevice(dev_idx);
  allocDevCtrl(&ctrl_buff);
  CUDACHECK(cudaEventCreate(&sync_event));

  sock_task_queues.reserve(n_data_socks);
  for (int i = 0; i < n_data_socks; ++i) {
    sock_task_queues.push_back(new SocketTaskQueue());
  }

  // at most has N_HOST_MEM_SLOTS requests
  for (int i = 0; i < N_HOST_MEM_SLOTS; ++i) {
    completed_requests.push(new SocketRequest());
  }

  int max_socket_tasks = n_data_socks * N_HOST_MEM_SLOTS;
  for (int i = 0; i < max_socket_tasks; ++i) {
    completed_tasks.push(new SocketTask());
  }

}

static inline void sendNetConnectionInfo(int& rank, ConnectionType_t type, int& fd) {
  PeerConnectionInfo my_info;
  my_info.peer_rank = rank;
  my_info.conn_type = type;
  int ret = ::send(fd, &my_info, sizeof(PeerConnectionInfo), 0);
  LOG_IF_ERROR(ret != sizeof(PeerConnectionInfo),
               "send connection info to peer failed, ret val %d, %s", ret, strerror(errno));
}

int recvNewListenPort(int& fd) {
  int port;
  auto ret = ::recv(fd, &port, sizeof(int), 0);
  LOG_IF_ERROR(ret != sizeof(int), "recv new listen port failed");
  return port;
}

static inline void buildNetSendDataConns(int* ip, int port, int n, std::vector<int>& data_fds) {
  for (int i = 0; i < n; ++i) {
    int fd = createSocketClient(ip, port, true);
    LOG_IF_ERROR(fd < 0, "create net send data connection failed");
    data_fds.push_back(fd);
  }
}

void NetConnection::launchBackgroundThreads() {
  for (int i = 0; i < n_data_socks; ++i) {
    background_threads.emplace_back(persistentSocketThread, this, i,
                                    sock_task_queues[i]);
  }
}

void NetConnection::persistentSocketThread(NetConnection* conn, int tid, SocketTaskQueue* task_queue) {

  SocketTask* task = nullptr;
  int ret = -1;
  while (!conn->close) {
    if (!task_queue->tasks.empty()) {
      {
        std::lock_guard<std::mutex> lk(task_queue->q_lock);
        task = task_queue->tasks.front();
        task_queue->tasks.pop();
      }
      LOG_IF_ERROR(task->stage != 0, "using a socket task stage != 0");
      if (task->is_send) {
        ret = ::send(task->fd, task->ptr, task->size, MSG_WAITALL);
        LOG_IF_ERROR(ret != task->size, "send socket task failed");
      } else {
        ret = ::recv(task->fd, task->ptr, task->size, MSG_WAITALL);
        LOG_IF_ERROR(ret != task->size, "recv socket task failed");
      }

      task->stage = 2;
      task_queue->completion_count++;
    }
  }
}

NetConnection::NetConnection(NetSendConnArgs& args,
                             int dev_idx,
                             int n_cuda_threads)
    : n_cuda_threads(n_cuda_threads),
      is_send(true),
      dev_idx(dev_idx),
      n_data_socks(args.n_socks) {
  initBuffer();
  ctrl_fd = createSocketClient(args.peer_ip, args.peer_port, true);
  sendNetConnectionInfo(args.self_rank, Net, ctrl_fd);
  int new_port = recvNewListenPort(ctrl_fd);
  data_fds.reserve(args.n_socks);
  buildNetSendDataConns(args.peer_ip, new_port, args.n_socks, this->data_fds);

  launchBackgroundThreads();
}

static inline void sendNewListenPort(int& ctrl_fd, int& new_listen_fd) {
  int new_listen_port;
  getSocketPort(&new_listen_fd, &new_listen_port);
  auto ret = ::send(ctrl_fd, &new_listen_port, sizeof(int), 0);
  LOG_IF_ERROR(ret != sizeof(int), "send new listen port failed");
  LOG_DEBUG("Sent new listen port %d", new_listen_port);
}

static inline void buildNetRecvDataConns(int& listen_fd, int& n, std::vector<int>& data_fds) {
  for (int i = 0; i < n; ++i) {
    int fd = socketAccept(listen_fd, true);
    LOG_IF_ERROR(fd < 0, "create net data recv connection failed");
    data_fds.push_back(fd);
    // LOG_DEBUG("Get a data recv sock fd %d (%d/%d)", fd, i+1, n);
  }
}

NetConnection::NetConnection(NetRecvConnArgs& args,
                             int dev_idx,
                             int n_cuda_threads)
    : n_cuda_threads(n_cuda_threads),
      is_send(false),
      dev_idx(dev_idx),
      n_data_socks(args.n_socks) {
  initBuffer();
  LOG_DEBUG("Net connection (recv) build buffer initialized");
  ctrl_fd = args.ctrl_fd;
  createListenSocket(&listen_fd, 0);
  LOG_DEBUG("Created new listen port for socket data connections");
  sendNewListenPort(ctrl_fd, listen_fd);
  buildNetRecvDataConns(listen_fd, args.n_socks, this->data_fds);

  launchBackgroundThreads();
}

void resetCtrlStatus(hostDevShmInfo* ctrl_buff) {
  ctrl_buff->head = N_HOST_MEM_SLOTS;
  ctrl_buff->tail = 0;
  ctrl_buff->size_idx = 0;
  memset(ctrl_buff->size_fifo, 0, sizeof(ctrl_buff->size_fifo));
}

SocketRequest* NetConnection::launchSocketRequest(void* ptr, int size) {
  if (completed_requests.empty()) return nullptr;

  SocketRequest* req = completed_requests.front();
  completed_requests.pop();
  LOG_IF_ERROR(req->stage != 0, "request is not reset to stage 0");

  int chunk_offset = 0, i = 0;
  int sub_size = std::max(SOCK_MIN_SIZE, DIVUP(size, n_data_socks));

  while (chunk_offset < size) {
    int real_size = std::min(sub_size, size - chunk_offset);
    SocketTask* t;
    if (completed_tasks.empty()) return nullptr; // not able to launch tasks for now

    {
      std::lock_guard<std::mutex> lk(this->completed_task_mtx);
      t = completed_tasks.front();
      completed_tasks.pop();
    }

    LOG_IF_ERROR(t->stage != 0, "using a task slot that is not in stage 0");
    t->fd = data_fds[i];
    t->is_send = this->is_send;
    t->offset = 0;
    t->size = real_size;
    t->ptr = (char*)ptr + chunk_offset;
    t->stage = 1;

    chunk_offset += real_size;
    req->sub_tasks.push_back(t);

    i++;
  }
  req->n_sub = i;
  req->size = size;
  req->stage = 1;

  for (SocketTask* t : req->sub_tasks) {
    std::lock_guard<std::mutex> lk(sock_task_queues[i]->q_lock);
    sock_task_queues[i]->tasks.push(t);
  }

  return req;
}

bool NetConnection::isRequestDone(SocketRequest* req) {
  if (req->stage == 0) {
    LOG_ERROR("checking on a request has stage 0");
    return false;
  }

  int n_comp = 0;
  for (int i = 0; i < req->n_sub; ++i) {
    if (req->sub_tasks[i]->stage == 2)
      n_comp++;
  }
  if (n_comp == req->n_sub)
    return true;

  return false;
}

// int NetConnection::executeWait(void* ptr, int nbytes, bool is_send) {
//   // find a empty request slot
//   int req_slot_idx = -1;
//   for (int i = 0; i < N_HOST_MEM_SLOTS; ++i) {
//     if (requests[i].stage == 0) {
//       req_slot_idx = i;
//       break;
//     }
//   }
//   if (req_slot_idx == -1) {
//     LOG_ERROR("cannot find empty socket request slot, return -1");
//     return -1;
//   }

//   launchSocketRequest(req_slot_idx, ptr, nbytes);
//   while (!isRequestDone(req_slot_idx)) {
//     std::this_thread::sleep_for(std::chrono::microseconds(100));
//   }

//   // clear status
//   requests[req_slot_idx].stage = 0;
//   for (int i = 0; i < requests[req_slot_idx].n_sub; ++i) {
//     requests[req_slot_idx].sub_tasks[i]->stage = 0;
//   }
//   return nbytes;
// }

void NetConnection::send(void* buff, size_t count, cudaStream_t stream) {
  // TODO: 
  // send the data size to peer, then launch netSendKernel
  // progressively launch socket tasks for parallel sending
  assert(is_send == true);

  int bytes = ::send(ctrl_fd, &count, sizeof(count), 0);
  if (bytes != sizeof(count)) {
    LOG_ERROR("sending msg size failed");
    return;
  }

  // launch kernel
  void* kernel_args[3] = {&buff, &ctrl_buff, &count};
  CUDACHECK(cudaLaunchKernel((void*)netSendKernel, dim3(1), dim3(n_cuda_threads), kernel_args, 0, stream));
  CUDACHECK(cudaEventRecord(sync_event, stream));

  // init variables
  size_t offset = 0;
  int slot_idx = 0; // equal to ctrl_buff->size_idx
  int next_req_slot = 0;
  size_t last_tail = 0;

  while (offset < count) {
    if (ctrl_buff->head < ctrl_buff->tail + N_HOST_MEM_SLOTS) {
      int real_size = ctrl_buff->size_fifo[slot_idx];
      // FIXME: for now only use first data socket with sync socket operation
      // int send_bytes = ::send(data_fds[0], ctrl_buff->ptr_fifo[slot_idx], real_size, 0);
      int send_bytes = executeWait(ctrl_buff->ptr_fifo[slot_idx], real_size, this->is_send);
      LOG_IF_ERROR(send_bytes != real_size, "send data via sock fd %d failed", data_fds[0]);
      // LOG_DEBUG("send size %d, fd %d", real_size, data_fds[0]);

      ctrl_buff->size_fifo[slot_idx] = 0;
      ctrl_buff->head++;
      offset += real_size;
      slot_idx = (slot_idx + 1) % N_HOST_MEM_SLOTS;
    }
  }
  CUDACHECK(cudaEventSynchronize(sync_event));
  resetCtrlStatus(ctrl_buff);

}

void NetConnection::recv(void* buff, size_t count, cudaStream_t stream) {
  // TODO:
  // recv the size from peer connection and verify
  assert(is_send == false);

  size_t recv_signal;
  size_t bytes = ::recv(ctrl_fd, &recv_signal, sizeof(count), 0);
  LOG_IF_ERROR(bytes != sizeof(size_t), "receeving message size from peer failed");
  if (recv_signal != count) {
    LOG_ERROR("received message size %lu does not match launched size %lu", recv_signal, count);
    return;
  }
  LOG_DEBUG("net connection recv, receiving %lu", recv_signal);

  // launch cuda kernel
  void* kernel_args[3] = {&buff, &ctrl_buff, &count};
  CUDACHECK(cudaLaunchKernel((void*)netRecvKernel, dim3(1), dim3(n_cuda_threads), kernel_args, 0, stream));
  CUDACHECK(cudaEventRecord(sync_event, stream));
  // LOG_DEBUG("Launched netRecvKernel, ptr %p, size %lu", buff, count);

  // launch socket requests 
  // init variables
  size_t offset = 0; // denotes completed offset
  size_t socket_offset = 0; // denotes ongoing socket offset
  int slot_idx = 0; // equal to ctrl_buff->size_idx
  volatile size_t* tail_ptr = &ctrl_buff->tail;

  while (offset < count) {
    if (ctrl_buff->head > ctrl_buff->tail) {
      // there is memory slots
      int chunk_size = MEM_SLOT_SIZE;
      if (count - offset < MEM_SLOT_SIZE) {
        // fill the full memory buff
        chunk_size = count - offset;
      }
      // FIXME: only use first data sock for recv
      // int recv_bytes = ::recv(data_fds[0], ctrl_buff->ptr_fifo[slot_idx], chunk_size, MSG_WAITALL);
      int recv_bytes = this->executeWait(ctrl_buff->ptr_fifo[slot_idx], chunk_size, this->is_send);
      LOG_IF_ERROR(
          recv_bytes != chunk_size,
          "receive data from sock %d failed, ret %d, chunksize %d, err %s",
          data_fds[0], recv_bytes, chunk_size, strerror(errno));

      ctrl_buff->size_fifo[slot_idx] = chunk_size;
      slot_idx = (slot_idx+1) % N_HOST_MEM_SLOTS;
      offset += chunk_size;
      // ctrl_buff->tail++;
      (*tail_ptr)++;
    }
  }
  CUDACHECK(cudaEventSynchronize(sync_event));
  resetCtrlStatus(ctrl_buff);
  // // reset socketTasks
  // for (int i = 0; i < n_data_socks; ++i) {
  //   task_queue[i].head = 0;
  //   task_queue[i].tail = 0;
  //   memset(task_queue[i].tasks, 0, sizeof(SocketTask) * N_HOST_MEM_SLOTS);
  // }
  
}


NetConnection::~NetConnection() {
  close = true;
  for (auto& t : background_threads) {
    if (t.joinable()) t.join();
  }

  while (!completed_requests.empty()) {
    delete completed_requests.front();
    completed_requests.pop();
  }

  while (!completed_tasks.empty()) {
    delete completed_tasks.front();
    completed_tasks.pop();
  }

  for (int i = 0; i < n_data_socks; ++i) {
    delete sock_task_queues[i];
  }

  freeDevCtrl(ctrl_buff);
}

P2PConnection::P2PConnection(P2PSendArgs& args) {
  int local_ip[4] = {127,0,0,1};
  ctrl_fd = createSocketClient(local_ip, args.peer_port, true);
  // send 
  PeerConnectionInfo conn_info;
  conn_info.peer_rank = args.self_rank;
  conn_info.conn_type = P2P;
  LOG_IF_ERROR(
      ::send(ctrl_fd, &conn_info, sizeof(conn_info), 0) != sizeof(conn_info),
      "p2p sending connection info failed");

}

P2PConnection::P2PConnection(P2PRecvArgs& args) {
  ctrl_fd = args.ctrl_fd;
  self_rank = args.self_rank;
}

ConnectionType_t P2PConnection::getType() {
  return P2P;
}

void P2PConnection::commonInit() {
  n_cuda_threads = N_CUDA_THREADS;
  CUDACHECK(cudaEventCreate(&sync_event));
}

void P2PConnection::send(void* buff, size_t nbytes, cudaStream_t stream) {
  // send the size info for mutual agree
  LOG_IF_ERROR(::send(ctrl_fd, &nbytes, sizeof(nbytes), 0) != sizeof(nbytes),
               "p2p sending size confirmation failed");

  // recv the memory handle from receiver
  cudaIpcMemHandle_t ipc_handle;
  LOG_IF_ERROR(
      ::recv(ctrl_fd, &ipc_handle, sizeof(ipc_handle), 0) != sizeof(ipc_handle),
      "p2p receiving ipc handle failed");
  
  void* dst_ipc_ptr;
  CUDACHECK(cudaIpcOpenMemHandle(&dst_ipc_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  // launch cuda kernel for data movement
  void* kernel_args[3] = {&dst_ipc_ptr, &buff, &nbytes};
  CUDACHECK(cudaLaunchKernel((void*)p2pSendKernel, dim3(1), dim3(n_cuda_threads), kernel_args, 0, stream));
  CUDACHECK(cudaEventRecord(sync_event, stream));

  // wait for completion
  CUDACHECK(cudaEventSynchronize(sync_event));
  // ack to peer
  int ack = 1;
  LOG_IF_ERROR(::send(ctrl_fd, &ack, sizeof(ack), 0) != sizeof(ack),
               "p2p sending ack failed");
}

void P2PConnection::recv(void* buff, size_t nbytes, cudaStream_t stream) {
  size_t src_size;
  ::recv(ctrl_fd, &src_size, sizeof(size_t), 0);
  if (src_size != nbytes) {
    LOG_ERROR("p2p received data size != launched size");
    return;
  }

  cudaIpcMemHandle_t ipc_handle;
  CUDACHECK(cudaIpcGetMemHandle(&ipc_handle, buff));
  LOG_IF_ERROR(
      ::send(ctrl_fd, &ipc_handle, sizeof(ipc_handle), 0) != sizeof(ipc_handle),
      "p2p receiver sending ipc handle failed");
  
  // wait for completion
  int complete;
  LOG_IF_ERROR(::recv(ctrl_fd, &complete, sizeof(complete), 0) != sizeof(int),
               "p2p receiver waiting for completion failed");
}
