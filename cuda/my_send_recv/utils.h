#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cstring>

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
void hostAlloc(T** ptr, size_t nelem) {
  CUDACHECK(cudaHostAlloc(ptr, nelem * sizeof(T), cudaHostAllocMapped));
  memset(*ptr, 0, nelem * sizeof(T));
}

template<typename T>
void fillVals(T* buff, size_t count) {
  for (int i = 0; i < count; ++i) {
    T e = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    buff[i] = e;
  }
}

void hostFree(void* ptr);

double timeMs();

uint64_t getHash(const char* string, int n);

bool getHostName(char* hostname, int maxlen, const char delim);

uint64_t getHostHash(void);

void ipStrToInts(std::string& ip, int* ret);

bool createListenSocket(int* fd, int port);

void getSocketPort(int* fd, int* port);