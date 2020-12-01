#ifndef __KERNS_H_
#define __KERNS_H_

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

#endif