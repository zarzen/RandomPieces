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

template<typename T>
static void fillVals(T* buff, size_t count) {
  for (int i = 0; i < count; ++i) {
    T e = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    buff[i] = e;
  }
}

static inline void hostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
}

static inline double timeMs() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count() /
         1e6;
};