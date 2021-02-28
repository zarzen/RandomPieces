#pragma once
#include <cstdlib>
#include <vector>
#include <thread>
#include "common_structs.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <queue>
#include <mutex>

typedef enum { Net = 0, P2P = 1, SHM = 2} ConnectionType_t;

// network, shm, p2p through this interface
class Connection {
 public:
  // for GPU buffers, launch different kernels
  virtual void send(void* buff, size_t nbytes, cudaStream_t stream) = 0;
  // for GPU buffers
  virtual void recv(void* buff, size_t nbytes, cudaStream_t stream) = 0;

  virtual ConnectionType_t getType() = 0;
};

struct NetSendConnArgs {
  int self_rank;
  int peer_ip[4];
  int peer_port;
  int n_socks;
  int n_threads;
};

struct NetRecvConnArgs {
  int self_rank;
  int ctrl_fd;
  int n_socks; // num of data socks
  int n_threads;
};

struct PeerConnectionInfo {
  int peer_rank;
  ConnectionType_t conn_type;
};

struct SocketTask {
  void* ptr;
  int offset; 
  int size; 
  int fd; 
  bool is_send; // 
  int stage; // 0: not launched; 1: ready to operate on; 2: completed
};

// task queue for each socket fd
// at most have N_HOST_MEM_SLOTS ongoing task for a socket
struct SocketTaskQueue {
  std::mutex q_lock;
  std::queue<SocketTask*> tasks;
  size_t completion_count;

  SocketTaskQueue() : completion_count(0) {}
};

struct SocketRequest {
  int stage; // 0: empty for occupancy; 1: launched
  std::vector<SocketTask*> sub_tasks;
  int n_sub;
  int size;
  SocketRequest() : stage(0), n_sub(0), size(0) {}
};

// TCP network connection
class NetConnection : public Connection {
  bool close;
  int n_cuda_threads;
  cudaEvent_t sync_event;

  int n_data_socks; // each sock has a thread 

  std::vector<int> data_fds;
  std::vector<SocketTaskQueue*> sock_task_queues;
  std::queue<SocketTask*> completed_tasks;
  std::mutex completed_task_mtx;

  int ctrl_fd;
  bool is_send;
  int dev_idx;
  hostDevShmInfo* ctrl_buff;

  int listen_fd; // only used for recv connection
  void initBuffer();

  // all operation on main threads, no lock
  std::queue<SocketRequest*> completed_requests; // save the pre-allocated socketRequest, avoid dynamic new
  std::queue<SocketRequest*> ongoing_requests; // ongoing request, check for completion

  std::vector<std::thread> background_threads;
  void launchBackgroundThreads();
  static void persistentSocketThread(NetConnection* conn, int tid, SocketTaskQueue* task_queue);

  // int executeWait(void* ptr, int nbytes, bool is_send);
  SocketRequest* launchSocketRequest(void* ptr, int size);
  bool isRequestDone(SocketRequest* req);
  void resetRequest(SocketRequest* req);
  
 public:
  NetConnection(NetSendConnArgs& args, int dev_idx, int n_cuda_threads);
  NetConnection(NetRecvConnArgs& args, int dev_idx, int n_cuda_threads);

  // sync operations
  void send(void* buff, size_t nbytes, cudaStream_t stream);
  void recv(void* buff, size_t nbytes, cudaStream_t stream);

  ConnectionType_t getType();
  ~NetConnection();
};

struct P2PSendArgs {
  int self_rank;
  int peer_port; // build tcp connection to same host
};

struct P2PRecvArgs {
  int self_rank;
  int ctrl_fd;
};
class P2PConnection : public Connection {
  int self_rank;
  int ctrl_fd;
  
  int n_cuda_threads;
  cudaEvent_t sync_event;
  void commonInit();
public:
  P2PConnection(P2PSendArgs& args);
  P2PConnection(P2PRecvArgs& args);

  void send(void* buff, size_t nbytes, cudaStream_t stream);
  void recv(void* buff, size_t nbytes, cudaStream_t stream);

  ConnectionType_t getType();
};
