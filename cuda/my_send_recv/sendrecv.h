#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <thread>
#include "rendezvous.h"
#include "connection.h"

using std::unordered_map;
using std::mutex;
using std::thread;

#define N_CUDA_THREADS 320 // must be multiple of 32 (warp_size)
#define N_HOST_MEM_SLOTS 4
#define MEM_SLOT_SIZE 1048576  // in bytes; must be a multiple of 16 (128bits).
#define CACHE_LINE_SIZE 128

#define MAX_HANDLE_N 4294967295
typedef unsigned long handle_t;

// exchange msg with host
struct hostDevShmInfo {
  // index of fifo to get item
  int size_idx = 0;

  int size_fifo[N_HOST_MEM_SLOTS] = {0};
  // pre allocated memory buffers on Host
  void* ptr_fifo[N_HOST_MEM_SLOTS] = {nullptr};
  size_t head = N_HOST_MEM_SLOTS; // consumer increase head
  size_t tail = 0;
  // consider padding it later
};

void initTaskInfo(hostDevShmInfo** info);

struct CommunicatorArgs {
  int rendezvous_ip[4];
  int rendezvous_port;
  int rank;
  int nranks;
  int dev_idx;
  int local_ip[4]; // ip of current node, consider remove later
};

struct CommunicationTask {
  void* dev_ptr;
  int peer;
  size_t bytes;
  handle_t handle;
  CommunicationTask(void* _ptr,
                    int _peer,
                    size_t _bytes,
                    handle_t _handle)
      : dev_ptr(_ptr),
        peer(_peer),
        bytes(_bytes),
        handle(_handle) {}
};

// record context informations
//
class Communicator {
  int N_SOCKET = 16;
  int N_SOCKET_THREAD = 2;

  struct peerInfo self_info;
  RendezvousServer* rendez_server;
  RendezvousClient* rendez_client;

  handle_t handle_num;
  mutex handle_mtx;
  handle_t getHandle();

  unordered_map<int, Connection*> send_conns;
  unordered_map<int, Connection*> recv_conns;

  std::queue<CommunicationTask> send_tasks;
  mutex send_task_mtx;
  std::queue<CommunicationTask> recv_tasks;
  mutex recv_task_mtx;

  struct hostDevShmInfo* recv_ctrl_buff;
  struct hostDevShmInfo* send_ctrl_buff;
  // 
  void initBuffers(CommunicatorArgs& args);

  void initLocally(CommunicatorArgs& args, int& listen_fd);
  //
  void initThreads(CommunicatorArgs& args, int& listen_fd);
  
  // receive connections are passively built
  Connection* buildConnection(int peer, bool is_send);
  Connection* getConnection(int peer, bool is_send);
  handle_t enqueueTask(std::queue<CommunicationTask>& task_queue,
                      mutex& mtx,
                       void*& buff,
                       size_t& bytes,
                       int& peer);

  static void persistentThreadListen(Communicator* comm, int fd);
  std::thread peer_listen_thd;
  static void persistentThreadSend(Communicator* comm, std::queue<CommunicationTask>& task_queue);
  std::thread send_dispatch_thd;
  static void persistentThreadRecv(Communicator* comm, std::queue<CommunicationTask>& task_queue);
  std::thread recv_dispatch_thd;

 public:
  Communicator(CommunicatorArgs& args);
  ~Communicator();

  handle_t enqueueSendTask(void* buff, size_t bytes, int peer);
  handle_t enqueueRecvTask(void* buff, size_t bytes, int peer);

  bool isCompleted(handle_t& handle);
};

// change name later
// async
handle_t ourSend(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm);

handle_t ourRecv(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm);

void waitTask(handle_t& handle, Communicator& comm);