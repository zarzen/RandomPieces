#pragma once
#include <cstdlib>
#include <vector>
#include <thread>
#include "common_structs.h"
#include <cuda_runtime.h>
#include <cuda.h>

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
  int stage; // 0: not launched; 1: ready to operate on
};

// task queue for each socket fd
// at most have N_HOST_MEM_SLOTS ongoing task for a socket
struct SocketTaskQueue {
  size_t head;
  size_t tail;
  SocketTask tasks[N_HOST_MEM_SLOTS];
  SocketTaskQueue():head(N_HOST_MEM_SLOTS), tail(0) {}
};

struct SocketRequest {
  int stage; // 0: empty for occupancy; 1: launched
  SocketTask** sub_tasks; // at most n_data_socks 
  int n_sub;
  size_t size;
  int slot_idx; // for debugging
};

// TCP network connection
class NetConnection : public Connection {
  bool close;
  int n_cuda_threads;
  cudaEvent_t sync_event;

  int n_data_socks;
  int n_threads;
  std::vector<int> data_fds;
  int ctrl_fd;
  bool is_send;
  int dev_idx;
  hostDevShmInfo* ctrl_buff;

  int listen_fd; // only used for recv connection
  void initBuffer();

  SocketRequest requests[N_HOST_MEM_SLOTS]; // at most that many socket request for a connection
  SocketTaskQueue* task_queue;
  std::vector<std::thread> background_threads;
  void launchBackgroundThreads();
  static void persistentSocketThread(NetConnection* conn, int tid, SocketTaskQueue* task_queue);

  void launchSocketRequest(int idx, void* ptr, int size);
  bool isRequestDone(int idx);
 public:
  NetConnection(NetSendConnArgs& args, int dev_idx, int n_cuda_threads);
  NetConnection(NetRecvConnArgs& args, int dev_idx, int n_cuda_threads);

  // sync operations
  void send(void* buff, size_t nbytes, cudaStream_t stream);
  void recv(void* buff, size_t nbytes, cudaStream_t stream);

  ConnectionType_t getType();
  ~NetConnection();
};

class P2PConnection : public Connection {

public:
  P2PConnection();
  void send(void* buff, size_t nbytes, cudaStream_t stream);
  void recv(void* buff, size_t nbytes, cudaStream_t stream);

  ConnectionType_t getType();
};
