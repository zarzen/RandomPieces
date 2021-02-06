#include "kernels.h"
#include <stdio.h>
#include <device_launch_parameters.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

inline __device__ void waitUntilN(TestControl* c, int n) {
    volatile int* signalPtr = &c->s;
    int signalCache = *signalPtr;
    while (signalCache != n) {
        signalCache = *signalPtr;
    }
    printf("[%d, %d] wait %d completed\n", blockIdx.x, threadIdx.x, n);
    char* d = c->buff + threadIdx.x * 8;
    (*(int*)d)++;
}

__global__ void 
waitSignal(TestControl* c) {
    printf("kernel pointer testcontrol %p\n", c);
    printf("enter kernel, %d\n", c->s);
    waitUntilN(c, 2);

    waitUntilN(c, 4);
}


void launchWait(TestControl* c) {
    void* args[1] = {&c};
    // waitSignal<<<dim3(1), dim3(2)>>>(c);
    cudaLaunchKernel((void*)waitSignal, dim3(1), dim3(2), args, 0, NULL);
}

template <class T>
__global__ static void SumKernel(T* b1, T* b2, size_t nelem, int gridx, int blockx) {
  // int index = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthd = gridx * blockx;
  size_t thread_step = nelem / nthd;
  size_t offset = thread_step * tid;
  size_t remain = nelem - thread_step * nthd;
  if (tid == nthd - 1) thread_step += remain;
  for (int i = 0; i < thread_step; ++i) {
    if (offset + i >= nelem) return;
    b1[offset+i] += b2[offset+i];
  }
}
#define BLOCK 640
inline dim3 cuda_gridsize_1d(int n) {
  int x = (n - 1) / BLOCK + 1;
  dim3 d = {(uint)x, 1, 1};
  return d;
}

void sumTwoBufferToFirst(void* b1, void* b2, size_t count, cudaStream_t stream) {
  // dim3 grid = {2,1,1};
  SumKernel<float><<<2, BLOCK, 0, stream>>>(
      (float*)b1, (float*)b2, count, 2, BLOCK);
}

void StreamCreate(cudaStream_t *stream){
  int greatest_priority;
  cudaError_t err;
  err = cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority);
  if (err != cudaSuccess) {
    printf("error happend while cudaDeviceGetStreamPriorityRange\n");
  }
  err = cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority);
  if (err != cudaSuccess) {
    printf("error while cudaStreamCreateWithPriority\n");
  }
}