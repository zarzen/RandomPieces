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

  task_queue = new SocketTaskQueue[n_data_socks];

  for (int i = 0; i < N_HOST_MEM_SLOTS; ++i) {
    requests[i].sub_tasks = new SocketTask*[n_data_socks];
    memset(requests[i].sub_tasks, 0, sizeof(SocketTask*) * n_data_socks);
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
  for (int i = 0; i < n_threads; ++i) {
    background_threads.emplace_back(persistentSocketThread, this, i,
                                    task_queue);
  }
}

void NetConnection::persistentSocketThread(NetConnection* conn, int tid, SocketTaskQueue* task_queue) {
  // TODO: 
  int n_sock_per_thread = conn->n_data_socks / conn->n_threads;
  SocketTaskQueue* my_queues = task_queue + tid * n_sock_per_thread;

  while (!conn->close) {
    for (int i = 0; i < n_sock_per_thread; ++i) {
      SocketTaskQueue* sock_queue = my_queues + i;
      LOG_IF_ERROR(sock_queue == NULL, "SocketTaskQueue is nullptr");

      int task_idx = sock_queue->head % N_HOST_MEM_SLOTS;
      SocketTask* t = &sock_queue->tasks[task_idx];
      LOG_IF_ERROR(t == NULL, "SocketTask t is nullptr");

      if (t->stage == 1 && t->offset < t->size) {
        // LOG_DEBUG("working on [%s] fd %d, task %d, stage %d, offset %d, ptr %p, size %d", 
        //     conn->is_send ? "send": "recv", t->fd,
        //     task_idx, t->stage, t->offset, t->ptr, t->size);
        // progress on this task
        bool res = socketProgressOpt(t->is_send, t->fd, t->ptr, t->size, &t->offset, 0);
        LOG_IF_ERROR(!res, "socket progress failed");

        if (t->offset == t->size) {
          sock_queue->head++;  // consumer move, move to next task.
          // LOG_DEBUG("[%s] socket queue fd %d, idx %d, move head to %lu, tail %lu",
          //           t->is_send? "send": "recv",
          //           t->fd, task_idx, sock_queue->head, sock_queue->tail);
        }
      }
      
    }
  }
}

NetConnection::NetConnection(NetSendConnArgs& args,
                             int dev_idx,
                             int n_cuda_threads)
    : n_cuda_threads(n_cuda_threads),
      is_send(true),
      dev_idx(dev_idx),
      n_data_socks(args.n_socks),
      n_threads(args.n_threads) {
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
      n_data_socks(args.n_socks),
      n_threads(args.n_threads) {
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

void NetConnection::launchSocketRequest(int idx, void* ptr, int size) {
  if (requests[idx].stage != 0) {
    LOG_ERROR("launch socket request failed, there is ongoing request at slot %d", idx);
    return;
  }
  int chunk_offset = 0, i = 0;
  int sub_size = std::max(SOCK_MIN_SIZE, DIVUP(size, n_data_socks));
  while (chunk_offset < size) {
    int real_size = std::min(sub_size, size - chunk_offset);
    SocketTaskQueue* q = task_queue + i; // get the task queue for data_fds[i]
    int task_idx = q->tail % N_HOST_MEM_SLOTS;
    SocketTask* t = &q->tasks[task_idx];

    LOG_IF_ERROR(t->stage != 0, "using a task slot that is not in stage 0");
    t->fd = data_fds[i];
    t->is_send = this->is_send;
    t->offset = 0;
    t->size = real_size;
    t->ptr = (char*)ptr + chunk_offset;
    t->stage = 1;

    chunk_offset += real_size;
    requests[idx].sub_tasks[i] = t;
    i++;
    q->tail++;
  }
  requests[idx].n_sub = i;
  requests[idx].size = size;
  requests[idx].slot_idx = idx;
  requests[idx].stage = 1;
  LOG_DEBUG("launched socket requst: nsub %d, size %d, slot_idx %d, op %s", i,
            size, idx, is_send ? "send" : "recv");
}

bool NetConnection::isRequestDone(int idx) {
  if(requests[idx].stage == 0) return false;
  if (requests[idx].stage == 1) {
    int n_comp = 0;
    for (int i = 0; i < requests[idx].n_sub; ++i) {
      SocketTask* sub_t = requests[idx].sub_tasks[i];
      // LOG_DEBUG("sub_t %p, idx %d, i %d, n_sub %d", (void*)sub_t, idx, i, requests[idx].n_sub);
      if (sub_t->offset == sub_t->size) n_comp++;
    }
    if (n_comp == requests[idx].n_sub) return true;
  }
  return false;
}

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
  CUDACHECK(cudaLaunchKernel((void*)netSendKernel, dim3(1), dim3(n_cuda_threads), kernel_args, 0, NULL));
  CUDACHECK(cudaEventRecord(sync_event, NULL));

  // init variables
  size_t offset = 0;
  int slot_idx = 0; // equal to ctrl_buff->size_idx
  int next_req_slot = 0;
  size_t last_tail = 0;
  while (offset < count) {
    size_t cur_tail = ctrl_buff->tail;
    if (last_tail < cur_tail) {
      // item to send; as the producer added one to the queue since last time
      for (size_t i = last_tail; i < cur_tail; ++i) {
        // TODO: FIXME: pass in slot_idx: for occupy certain request slot;
        int real_size = ctrl_buff->size_fifo[next_req_slot];
        void* data_ptr = ctrl_buff->ptr_fifo[next_req_slot];
        LOG_DEBUG("i %lu, cur_tail %lu, data_ptr %p, size %d", i, cur_tail, data_ptr, real_size);

        launchSocketRequest(next_req_slot, data_ptr, real_size);
        next_req_slot = (next_req_slot + 1) % N_HOST_MEM_SLOTS;
      }
      last_tail = cur_tail;
    }

    // check a status of a request
    if (isRequestDone(slot_idx)) {
      LOG_DEBUG("completed [%s] request, of size %lu",
                is_send ? "send" : "recv", requests[slot_idx].size);

      offset += requests[slot_idx].size;
      // reset
      requests[slot_idx].stage = 0;
      for (int i = 0; i < requests[slot_idx].n_sub; ++i) {
        requests[slot_idx].sub_tasks[i]->stage = 0;
      }
      ctrl_buff->head++;
      slot_idx = (slot_idx + 1) % N_HOST_MEM_SLOTS;
    }

  }
  CUDACHECK(cudaEventSynchronize(sync_event));
  resetCtrlStatus(ctrl_buff);
  // reset socketTasks
  for (int i = 0; i < n_data_socks; ++i) {
    task_queue[i].head = 0;
    task_queue[i].tail = 0;
    memset(task_queue[i].tasks, 0, sizeof(SocketTask) * N_HOST_MEM_SLOTS);
  }
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
  LOG_DEBUG("Launched netRecvKernel, ptr %p, size %lu", buff, count);

  // launch socket requests 
  // init variables
  size_t offset = 0; // denotes completed offset
  size_t socket_offset = 0; // denotes ongoing socket offset
  int slot_idx = 0; // equal to ctrl_buff->size_idx
  int next_req_slot = 0;
  int ongoing_req = 0;
  size_t last_tail = 0;
  while (offset < count) {
    if (ongoing_req < N_HOST_MEM_SLOTS && socket_offset < count &&
        ctrl_buff->head > ctrl_buff->tail) {
      // has memory slots and there are remaining bytes to receive
      size_t real_size = std::min((size_t)MEM_SLOT_SIZE, count - socket_offset);
      void* data_ptr = ctrl_buff->ptr_fifo[next_req_slot];
      launchSocketRequest(next_req_slot, data_ptr, real_size);
      next_req_slot = (next_req_slot + 1) % N_HOST_MEM_SLOTS;
      socket_offset += real_size;
      ongoing_req++;
    }

    if (isRequestDone(slot_idx)) {
      LOG_DEBUG("completed [%s] request, of size %lu",
                is_send ? "send" : "recv", requests[slot_idx].size);

      offset += requests[slot_idx].size;
      requests[slot_idx].stage = 0;
      for (int i = 0; i < requests[slot_idx].n_sub; ++i) {
        requests[slot_idx].sub_tasks[i]->stage = 0;
      }
      // post size of the buffer
      ctrl_buff->size_fifo[slot_idx] = requests[slot_idx].size;
      ctrl_buff->tail++;
      slot_idx = (slot_idx + 1) % N_HOST_MEM_SLOTS;
      ongoing_req--;
    }

  }
  CUDACHECK(cudaEventSynchronize(sync_event));
  resetCtrlStatus(ctrl_buff);
  // reset socketTasks
  for (int i = 0; i < n_data_socks; ++i) {
    task_queue[i].head = 0;
    task_queue[i].tail = 0;
    memset(task_queue[i].tasks, 0, sizeof(SocketTask) * N_HOST_MEM_SLOTS);
  }
  
}


NetConnection::~NetConnection() {
  close = true;
  for (auto& t : background_threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < N_HOST_MEM_SLOTS; ++i) {
    delete[] requests[i].sub_tasks;
  }

  freeDevCtrl(ctrl_buff);
  delete[] task_queue;
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
