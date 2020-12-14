#ifndef __KERNS_H_
#define __KERNS_H_

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

struct TestControl {
    int s;
    char buff[128];
};

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