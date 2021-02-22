#include "sendrecv.h"
#include <cuda.h>
#include <chrono>
#include "logger.h"
#include "utils.h"

void initTaskInfo(hostDevShmInfo** info){
  hostAlloc<hostDevShmInfo>(info, 1);
  (*info)->tail = 0;
  (*info)->head = N_HOST_MEM_SLOTS;
  for (int i = 0; i < N_HOST_MEM_SLOTS; ++i) {
    void* pinned;
    CUDACHECK(cudaHostAlloc(&pinned, (MEM_SLOT_SIZE), cudaHostAllocMapped));
    (*info)->ptr_fifo[i] = pinned;
  }
}

void Communicator::initBuffers(CommunicatorArgs& args) {
  cudaSetDevice(args.dev_idx);
  initTaskInfo(&recv_ctrl_buff);
  initTaskInfo(&send_ctrl_buff);
}

/* start local peer listen socket and register the peer info */
void Communicator::initLocally(CommunicatorArgs& args, int& listen_fd) {
  rendez_client = new RendezvousClient(args.rendezvous_ip, args.rendezvous_port);

  int listen_port;
  createListenSocket(&listen_fd, 0);
  getSocketPort(&listen_fd, &listen_port);
  self_info.port = listen_port;
  self_info.rank = args.rank;
  self_info.nranks = args.nranks;
  self_info.dev_idx = args.dev_idx;
  memcpy(self_info.ip, args.local_ip, sizeof(int) * 4);
  self_info.host_hash = getHostHash();

  rendez_client->registerPeer(self_info);
}

void Communicator::initThreads(CommunicatorArgs& args, int& listen_fd) {
  // start the listen thread for peers

  // start send thread

  // start recv thread
}

Communicator::Communicator(CommunicatorArgs& args) {
  if (args.rank == 0) 
    rendez_server = new RendezvousServer(args.rendezvous_port, args.nranks);

  initBuffers(args);
  int listen_fd;
  initLocally(args, listen_fd);
  initThreads(args, listen_fd);
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
    }
    return found->second;
  } else {
    peerInfo peer_info = this->rendez_client->getPeerInfo(peer);
    // based on peer info to build corresponding connections
    if (self_info.host_hash != peer_info.host_hash) {
      // peer is at remote -> build NetConnection
      NetSendConnArgs args;
      args.n_socks = N_SOCKET;
      args.n_threads = N_SOCKET_THREAD;
      args.peer_ip = peer_info.ip;
      args.peer_port = peer_info.port;
      NetConnection* peer_conn = new NetConnection(args);
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
  return ret;
}

handle_t Communicator::enqueueSendTask(void* buff, size_t bytes, int peer) {
  return enqueueTask(send_tasks, send_task_mtx, buff, bytes, peer);

}

handle_t Communicator::enqueueRecvTask(void* buff, size_t bytes, int peer) {
  return enqueueTask(recv_tasks, recv_task_mtx, buff, bytes, peer);
}

bool Communicator::isCompleted(handle_t& handle) {}

Communicator::~Communicator() {
  // clean resource
}

static void persistentThreadListen(Communicator* comm, int fd) {

}

static void persistentThreadSend(Communicator* comm, std::queue<CommunicationTask>& task_queue) {

}

static void persistentThreadRecv(Communicator* comm, std::queue<CommunicationTask>& task_queue) {

}

handle_t ourSend(void* dev_ptr,
                 size_t bytes_count,
                 int peer,
                 Communicator& comm) {
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