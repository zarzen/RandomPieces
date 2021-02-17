#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <unordered_map>

#define N_CUDA_THREADS 256
#define N_HOST_MEM_SLOTS 4
#define MEM_SLOT_SIZE 1024 * 1024  // in bytes
#define CACHE_LINE_SIZE 128

// network, shm, p2p through this interface
class Connection {
 public:
  // only for CPU buffers
  virtual void sendCtrl(void* buff, size_t count) = 0;
  // only for CPU buffers
  virtual void sendData(void* buff, size_t count) = 0;
};

// TCP network connection
class NetConnection : public Connection {};

// exchange msg with host
struct hostDevShmInfo {
  // index of fifo to get item
  int size_idx = 0;

  int size_fifo[N_HOST_MEM_SLOTS] = {0};
  // pre allocated memory buffers on Host
  void* ptr_fifo[N_HOST_MEM_SLOTS] = {nullptr};
  size_t head = 0;
  size_t tail = N_HOST_MEM_SLOTS;
  // consider padding it later
};

// record context informations
//
class Communicator {
 public:
  Communicator(std::string rendezvous_ip,
               int rendezvous_port,
               int rank,
               int nranks,
               int dev_idx);
  Connection& getConnection(int peer);
};

// change name later
void ourSend(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm);

void ourRecv(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm);