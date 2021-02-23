#pragma once

#include <stdint.h>

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
  CommunicationTask(){}
};

struct RankInfo {
  int rank;
  int nranks;
  int dev_idx;
  int ip[4]; // for connection building
  int port; // for connection building
  uint64_t host_hash;
};