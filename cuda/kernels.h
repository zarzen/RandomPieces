#ifndef __KERNS_H_
#define __KERNS_H_

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      e = cudaGetLastError(); \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      throw "cuda check failed";                                    \
    }                                                               \
  } while (0)

struct TestControl {
    int s;
    char buff[128];
};

#define BUFFER_SLOTS 4
#define SLOT_SIZE 1024*1024 // 1MB 
#define UNROLL 4
#define WARP_SIZE 32

struct CircularBuffer {
    int capacity = BUFFER_SLOTS;
    uint64_t head = 0;
    uint64_t tail = BUFFER_SLOTS;
    uint64_t consumer_head = 0; // head pointer of consumer
    void* data_buffers[BUFFER_SLOTS];
    uint32_t data_sizes[BUFFER_SLOTS];
    int count() {return capacity - (tail-head);};
};

struct Communicator {
    void* mem_for_send;
    void* mem_for_receive;
    
};

struct ControlContext {
    int peer; // currently not used
    Communicator* comm;
};

__global__ void sendDataOnGPU(const void* src, ControlContext* ctx);

__global__ void dataMove(const volatile float* src, volatile float* dst, size_t count);

typedef ulong2 Pack128;
typedef ulong4 Pack256;
__global__ void pack128Move(const Pack128* src, Pack128* dst, int count);

__global__ void pack256Move(const Pack256* src, Pack256* dst, size_t count);


__global__ void vectorAdd(const float* A,
                          const float* B,
                          float* C,
                          int numElements);


__global__ void waitSignal(TestControl* c);

inline __device__ void waitUntilN(TestControl* c, int n);

void launchWait(TestControl* c);

// assume float type
void sumTwoBufferToFirst(void* b1, void* b2, size_t count, cudaStream_t stream);

void StreamCreate(cudaStream_t *stream);
#endif