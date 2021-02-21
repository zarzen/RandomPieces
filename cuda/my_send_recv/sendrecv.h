#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <unordered_map>

#define N_CUDA_THREADS 320 // must be multiple of 32 (warp_size)
#define N_HOST_MEM_SLOTS 4
#define MEM_SLOT_SIZE 1048576  // in bytes; must be a multiple of 16 (128bits).
#define CACHE_LINE_SIZE 128

typedef unsigned long handle_t;

typedef enum { Net = 0, P2P = 1, SHM = 2} ConnectionType_t;

// network, shm, p2p through this interface
class Connection {
 public:
  // only for CPU buffers
  virtual void sendCtrl(void* buff, size_t count) = 0;
  // only for CPU buffers
  virtual void sendData(void* buff, size_t count) = 0;

  virtual ConnectionType_t getType() = 0;
};

// TCP network connection
class NetConnection : public Connection {
public:
  void sendCtrl(void* buff, size_t count);
  void sendData(void* buff, size_t count);

  ConnectionType_t getType();
};

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
  std::string rendezvous_ip;
  int rendezvous_port;
  int rank;
  int nranks;
  int dev_idx;
};

// record context informations
//
class Communicator {
 public:
  Communicator(CommunicatorArgs& context);
  Connection& getConnection(int peer);
};

// change name later
// async
handle_t ourSend(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm);

handle_t ourRecv(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm);

void waitTask(handle_t& handle, Communicator& comm);