#pragma once

#include "utils.h"
#include "sendrecv.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define DEFAULT_UNROLL 4
#define WARP_SIZE 32

typedef ulong2 Pack128;


__global__ void netSendKernel(void* send_buff, struct hostDevShmInfo* info, size_t count_bytes);
__global__ void shmSendKernel(struct hostDevShmInfo* info, size_t count);
// TODO: p2p copy kernel
__global__ void p2pSendKernel(struct hostDevShmInfo* info, size_t count);

__global__ void netRecvKernel(void* recv_buff, struct hostDevShmInfo* info, size_t count_bytes);
__global__ void shmRecvKernel(struct hostDevShmInfo* info, size_t count);
__global__ void p2pRecvKernel(struct hostDevShmInfo* info, size_t count);