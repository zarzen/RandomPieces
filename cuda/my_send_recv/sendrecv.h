#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <unordered_map>
#include <set>
#include <queue>
#include <thread>
#include "rendezvous.h"
#include "connection.h"
#include "common_structs.h"

using std::unordered_map;
using std::mutex;
using std::thread;

// record context informations
//
class Communicator {
  int N_SOCKET = 16;
  int N_SOCKET_THREAD = 2;

  bool shutdown;

  struct RankInfo self_info;
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

  void initLocally(CommunicatorArgs& args, int& listen_fd);
  //
  void initThreads(CommunicatorArgs& args, int& listen_fd);
  
  // receive connections are passively built
  Connection* buildConnection(int peer, bool is_send);
  Connection* getConnection(int peer, bool is_send);
  std::set<handle_t> ongoing_tasks;
  mutex ongoing_task_mtx;
  handle_t enqueueTask(std::queue<CommunicationTask>& task_queue,
                      mutex& mtx,
                       void*& buff,
                       size_t& bytes,
                       int& peer);
  void completeTask(handle_t& h);

  static void persistentThreadListen(Communicator* comm, int fd);
  std::thread peer_listen_thd;
  static void persistentThreadSend(Communicator* comm, mutex& mtx, std::queue<CommunicationTask>& task_queue);
  std::thread send_dispatch_thd;
  static void persistentThreadRecv(Communicator* comm, mutex& mtx, std::queue<CommunicationTask>& task_queue);
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