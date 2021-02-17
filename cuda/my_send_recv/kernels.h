#pragma once

#include "utils.h"
#include "sendrecv.h"

__global__ void netSendKernel(struct hostDevShmInfo* info, size_t count);
__global__ void shmSendKernel(struct hostDevShmInfo* info, size_t count);
__global__ void p2pSendKernel(struct hostDevShmInfo* info, size_t count);

__global__ void netRecvKernel(struct hostDevShmInfo* info, size_t count);
__global__ void shmRecvKernel(struct hostDevShmInfo* info, size_t count);
__global__ void p2pRecvKernel(struct hostDevShmInfo* info, size_t count);