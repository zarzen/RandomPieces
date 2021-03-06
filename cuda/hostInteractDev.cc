#include "kernels.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

// copied from nccl and modified for testing purpose only
/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclNumResults              =  6 } ncclResult_t;

template <typename T>
static ncclResult_t ncclCudaHostCalloc(T** ptr, size_t nelem) {
  CUDACHECK(cudaHostAlloc(ptr, nelem*sizeof(T), cudaHostAllocMapped));
  memset(*ptr, 0, nelem*sizeof(T));
  return ncclSuccess;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

void signalLoop(TestControl* c) {
    for (int i = 0; i < 10; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        c->s ++;
    }
}

void readSignalLoop(TestControl* c, int n, bool* exit) {
  int cached[n] = {0};
  while (!*exit) {
      for (int i = 0; i < n; ++i) {
        if (cached[i] != *(int*)(c->buff + i * 8)) {
          cached[i] = *(int*)(c->buff + i * 8);
          printf("cuda-thread-%d update val %d\n", i, cached[i]);
        }
      }
  }
}

// typedef void (*waitSignal_t)(TestControl* c) ;

bool accessibleAtHost(void* ptr, int device) {
  cudaSetDevice(device);

  cudaError_t err;
  cudaPointerAttributes attr;
  err = cudaPointerGetAttributes(&attr, ptr);
  switch (attr.type) {
    case cudaMemoryTypeUnregistered:
      return true;
    case cudaMemoryTypeHost:
      return true;
    default:
      return false;
  }
}

int main() {
    TestControl* c;
    ncclResult_t ret = ncclCudaHostCalloc(&c, 1);
    if (ret != ncclSuccess) {
        std::cerr << "allocation failed\n";
    }
    printf("pointer testcontrol %p\n", c);
    printf("Testcontrol pointer accessable at host <%s>\n", accessibleAtHost(c, 0)? "true": "false");
    void* hostMalloc = malloc(10);
    printf("Host malloc pointer accessable at host <%s>\n", accessibleAtHost(hostMalloc, 0)? "true": "false");
    void* cudaMallocPtr;
    cudaMalloc(&cudaMallocPtr, 10);
    printf("cudaMalloc pointer accessable at host <%s>\n", accessibleAtHost(cudaMallocPtr, 0)? "true": "false");
    
    c->s = 0;
    int nCudaThreads = 2;
    bool exit = false;
    std::thread signalThd(signalLoop, c);
    std::thread readValThd(readSignalLoop, c, nCudaThreads, &exit);
    // launch kernel
    void* args[1] = {&c};
    cudaLaunchKernel((void*)waitSignal, dim3(1), dim3(nCudaThreads), args, 0, NULL);
    
    printf("kernel launch complete\n");
    
    if (signalThd.joinable())
      signalThd.join();
    exit = true;
    readValThd.join();
    ncclCudaHostFree(c);
}