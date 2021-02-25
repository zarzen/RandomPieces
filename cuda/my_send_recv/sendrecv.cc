#include "sendrecv.h"
#include <cuda.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <chrono>
#include "logger.h"
#include "utils.h"

static void createStream(cudaStream_t* stream) {
  int greatest_priority;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
  CUDACHECK(cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority));
}

/* start local peer listen socket and register the peer info */
void Communicator::initLocally(CommunicatorArgs& args, int& listen_fd) {
  rendez_client = new RendezvousClient(args.rendezvous_ip, args.rendezvous_port);
  LOG_DEBUG("Created rendez-cli");

  send_stream = args.send_stream;
  recv_stream = args.recv_stream;
  if (send_stream == NULL) {
    createStream(&send_stream);
  }
  if (recv_stream == NULL) {
    createStream(&recv_stream);
  }

  int listen_port;
  createListenSocket(&listen_fd, 0);
  getSocketPort(&listen_fd, &listen_port);
  self_info.port = listen_port;
  self_info.rank = args.rank;
  self_info.nranks = args.nranks;
  self_info.dev_idx = args.dev_idx;
  memcpy(self_info.ip, args.local_ip, sizeof(int) * 4);
  self_info.host_hash = getHostHash();
  LOG_DEBUG(
      "Filled self info: rank %d, nranks %d, hostHash %lu, port %d, dev_idx %d",
      self_info.rank, self_info.nranks, self_info.host_hash, self_info.port,
      self_info.dev_idx);

  rendez_client->registerRankInfo(self_info);
  LOG_DEBUG("Rank %d registered self info to rendez-server", self_info.rank);
}

void Communicator::initThreads(CommunicatorArgs& args, int& listen_fd) {
  // start the listen thread for peers
  peer_listen_thd = std::thread(persistentThreadListen, this, listen_fd);
  // start send thread
  send_dispatch_thd = std::thread(persistentThreadSend, this, std::ref(send_task_mtx), std::ref(send_tasks));
  // start recv thread
  recv_dispatch_thd = std::thread(persistentThreadRecv, this, std::ref(recv_task_mtx), std::ref(recv_tasks));
}

Communicator::Communicator(CommunicatorArgs& args) {
  shutdown = false;
  if (args.rank == 0) {
    rendez_server = new RendezvousServer(args.rendezvous_port, args.nranks);
    LOG_DEBUG("Launched rendez-server at rank 0");
  }

  int listen_fd;
  initLocally(args, listen_fd);
  initThreads(args, listen_fd);
  getSocketPort(&listen_fd, &listen_port);
}

NetConnection* buildNetConnectionSend(RankInfo& self_info, RankInfo& peer_info, int& n_socks, int& n_threads, int& dev_idx) {
  NetSendConnArgs args;
  args.n_socks = n_socks;
  args.n_threads = n_threads;
  memcpy(args.peer_ip, peer_info.ip, sizeof(int) * 4);
  args.peer_port = peer_info.port;
  args.self_rank = self_info.rank;
  NetConnection* peer_conn = new NetConnection(args, dev_idx, N_CUDA_THREADS);
  return peer_conn;
}

Connection* Communicator::buildConnection(int peer, bool is_send) {
  unordered_map<int, Connection*>* conns;
  if (is_send)
    conns = &send_conns;
  else
    conns = &recv_conns;
  
  if (!is_send) {
    // recv links are passively built
    auto found = conns->find(peer);
    while (found == conns->end()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      found = conns->find(peer);
    }
    LOG_DEBUG("Found the receive connection type %d to peer %d",
              found->second->getType(), peer);
    return found->second;
  } else {
    RankInfo peer_info = this->rendez_client->getPeerInfo(peer);
    LOG_DEBUG("Got peer info: rank %d, listen-port %d, hosthash %lu",
              peer_info.rank, peer_info.port, peer_info.host_hash);
    // based on peer info to build corresponding connections
    if (self_info.host_hash != peer_info.host_hash) {
      // peer is at remote -> build NetConnection
      NetConnection* peer_conn =
          buildNetConnectionSend(this->self_info, peer_info, N_SOCKET,
                                 N_SOCKET_THREAD, self_info.dev_idx);
      send_conns[peer] = peer_conn;
      return peer_conn;
    } else {
      // same host
      int p2p;
      CUDACHECK(
          cudaDeviceCanAccessPeer(&p2p, self_info.dev_idx, peer_info.dev_idx));
      if (p2p == 0) {
        LOG_ERROR("p2p is not available between device %d and %d", self_info.dev_idx, peer_info.dev_idx);
        return nullptr;
      } else {
        LOG_DEBUG("p2p connection is built for device %d and %d", self_info.dev_idx, peer_info.dev_idx);
        LOG_ERROR("not implemented for p2p connections");
        return nullptr;
      }
    }
  }
}

Connection* Communicator::getConnection(int peer, bool is_send) {
  unordered_map<int, Connection*>* conns;
  if (is_send)
    conns = &send_conns;
  else
    conns = &recv_conns;

  auto found = conns->find(peer);
  if (found != conns->end()) {
    return found->second;
  } else {
    return this->buildConnection(peer, is_send);
  }
}

handle_t Communicator::getHandle() {
  handle_t ret;
  {
    std::lock_guard<mutex> lk(handle_mtx);
    ret = handle_num;
    handle_num++;
    if (handle_num >= MAX_HANDLE_N) handle_num = 0;
  }
  return ret;
}


handle_t Communicator::enqueueTask(std::queue<CommunicationTask>& task_queue,
                                   mutex& mtx,
                                   void*& buff,
                                   size_t& bytes,
                                   int& peer) {
  handle_t ret = getHandle();
  {
    std::lock_guard<mutex> lk(mtx);
    task_queue.emplace(buff, peer, bytes, ret);
  }
  {
    std::lock_guard<mutex> lk(ongoing_task_mtx);
    auto res = ongoing_tasks.insert(ret);
    LOG_IF_ERROR(res.first == ongoing_tasks.end(), "insert handle for ongoing task failed");
  }
  return ret;
}

void Communicator::completeTask(handle_t& h) {
  std::lock_guard<mutex> lk(ongoing_task_mtx);
  ongoing_tasks.erase(h);
}

handle_t Communicator::enqueueSendTask(void* buff, size_t bytes, int peer) {
  return enqueueTask(send_tasks, send_task_mtx, buff, bytes, peer);
}

handle_t Communicator::enqueueRecvTask(void* buff, size_t bytes, int peer) {
  // LOG_DEBUG("enqueue recv task, ptr %p, size %lu, peer %d", buff, bytes, peer);
  return enqueueTask(recv_tasks, recv_task_mtx, buff, bytes, peer);
}

// FIXME: it has problem when query with a handle that is not launched.
bool Communicator::isCompleted(handle_t& handle) {
  auto found = ongoing_tasks.find(handle);
  if (found == ongoing_tasks.end()) {
    return true;
  } else {
    return false;
  }
}

void Communicator::closeListenThread() {
  int local_ip[4] = {127,0,0,1};
  int tmp_fd = createSocketClient(local_ip, listen_port, true);
  PeerConnectionInfo tmp_info;
  tmp_info.peer_rank = -1;
  LOG_IF_ERROR(
      ::send(tmp_fd, &tmp_info, sizeof(tmp_info), 0) != sizeof(tmp_info),
      "closing listen thread failed");
}

Communicator::~Communicator() {
  shutdown = true;
  if (peer_listen_thd.joinable()) {
    // TODO: fix
    closeListenThread();
    peer_listen_thd.join();
  }
  if (send_dispatch_thd.joinable()) send_dispatch_thd.join();
  if (recv_dispatch_thd.joinable()) recv_dispatch_thd.join();

  if (args.rank == 0)
    delete rendez_server;
  delete rendez_client;
  for (auto p : recv_conns) {
    delete p.second;
  }
  for (auto p : send_conns) {
    delete p.second;
  }
}

void Communicator::persistentThreadListen(Communicator* comm, int fd) {
  // TODO: accept client fd, create correct Connection based on peer_info
  PeerConnectionInfo conn_info;

  while (!comm->shutdown) {
    int client_fd = socketAccept(fd, true);
    auto ret = ::recv(client_fd, &conn_info, sizeof(PeerConnectionInfo), MSG_WAITALL);
    LOG_IF_ERROR(ret != sizeof(PeerConnectionInfo), "recv connection protocol failed");

    if (conn_info.peer_rank == -1) return; // close signal

    if (conn_info.conn_type == Net) {
      LOG_DEBUG("Rank %d building connection type Net", comm->self_info.rank);
      NetRecvConnArgs net_recv_args;
      net_recv_args.ctrl_fd = client_fd;
      net_recv_args.n_socks = comm->N_SOCKET;
      net_recv_args.n_threads = comm->N_SOCKET_THREAD;
      NetConnection* conn = new NetConnection(net_recv_args, comm->self_info.dev_idx, N_CUDA_THREADS);
      comm->recv_conns[conn_info.peer_rank] = conn;
      LOG_DEBUG("Rank %d built a receive Net connection with rank %d",
                comm->self_info.rank, conn_info.peer_rank);
    }
    else if (conn_info.conn_type == P2P) {
      LOG_ERROR("Not implemented");
    }
    else if (conn_info.conn_type == SHM) {
      LOG_ERROR("Not implemented");
    }
  }
}

void Communicator::persistentThreadSend(Communicator* comm, mutex& mtx, std::queue<CommunicationTask>& task_queue) {
  CommunicationTask task;
  while (!comm->shutdown) {
    if (!task_queue.empty()) {
      {
        std::lock_guard<mutex> lk(mtx);
        task = task_queue.front();
        task_queue.pop();
      }
      Connection* conn = comm->getConnection(task.peer, true);
      LOG_DEBUG("comm send, ptr %p, size %lu", task.dev_ptr, task.bytes);

      conn->send(task.dev_ptr, task.bytes, comm->send_stream);
      comm->completeTask(task.handle);
    }
  }
}

void Communicator::persistentThreadRecv(Communicator* comm, mutex& mtx, std::queue<CommunicationTask>& task_queue) {
  CommunicationTask task;
  while (!comm->shutdown) {
    if (!task_queue.empty()) {
      {
        std::lock_guard<mutex> lk(mtx);
        task = task_queue.front();
        task_queue.pop();
      }
      Connection* conn = comm->getConnection(task.peer, false);
      LOG_DEBUG("Got the recv connection towards peer %d", task.peer);
      conn->recv(task.dev_ptr, task.bytes, comm->recv_stream);
      comm->completeTask(task.handle);
    }
  }
}

handle_t ourSend(void* dev_ptr,
                 size_t bytes_count,
                 int peer,
                 Communicator& comm) {
  // LOG_DEBUG("ourSend %p, size %lu, to peer %d", dev_ptr, bytes_count, peer);
  return comm.enqueueSendTask(dev_ptr, bytes_count, peer);
}

handle_t ourRecv(void* dev_ptr,
                 size_t bytes_count,
                 int peer,
                 Communicator& comm) {
  return comm.enqueueRecvTask(dev_ptr, bytes_count, peer);
}

void waitTask(handle_t& handle, Communicator& comm) {
  while (!comm.isCompleted(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}