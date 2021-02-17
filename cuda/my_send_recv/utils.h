#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      e = cudaGetLastError();                                       \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      throw "cuda check failed";                                    \
    }                                                               \
  } while (0)

// code copy and modified from nccl
template <typename T>
static void hostAlloc(T** ptr, size_t nelem) {
  CUDACHECK(cudaHostAlloc(ptr, nelem * sizeof(T), cudaHostAllocMapped));
  memset(*ptr, 0, nelem * sizeof(T));
}

static inline void hostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
}

static inline double timeMs() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count() /
         1e6;
};