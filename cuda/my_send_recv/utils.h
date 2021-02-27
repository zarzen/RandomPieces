#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include "logger.h"
#include "common_structs.h"
#include <sys/socket.h>

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

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
static inline void hostAlloc(T** ptr, size_t nelem) {
  CUDACHECK(cudaHostAlloc(ptr, nelem * sizeof(T), cudaHostAllocMapped));
  memset(*ptr, 0, nelem * sizeof(T));
}

template<typename T>
static inline void fillVals(T* buff, size_t count) {
  srand(123);
  for (int i = 0; i < count; ++i) {
    T e = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    buff[i] = e;
  }
}

void allocDevCtrl(hostDevShmInfo** info);

void freeDevCtrl(hostDevShmInfo* info);

void hostFree(void* ptr);

double timeMs();

uint64_t getHash(const char* string, int n);

bool getHostName(char* hostname, int maxlen, const char delim);

uint64_t getHostHash(void);

void ipStrToInts(std::string& ip, int* ret);

bool createListenSocket(int* fd, int port);

void getSocketPort(int* fd, int* port);

std::string getSocketIP(int& fd);

int socketAccept(int& server_fd, bool tcp_no_delay);

int createSocketClient(int* ip, int port, bool no_delay);

static bool socketProgressOpt(bool is_send, int fd, void* ptr, int size, int* offset, int block) {
  int bytes = 0;
  char* data = (char*)ptr;
  do {
    if (is_send) bytes = ::send(fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    else bytes = ::recv(fd, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
    
    if (!is_send && bytes == 0) {
      LOG_ERROR("Net : Connection closed by remote peer");
      return false;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        LOG_ERROR("Call to recv failed : %s", strerror(errno));
        return false;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
  } while (bytes > 0 && (*offset) < size);
  return true;
}

void printFloats(const char* prefix, float* buffer, int n);

double floatSummary(float* buff, int nelem);