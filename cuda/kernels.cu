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
    printf("[%d, %d] wait %d completed\n", blockIdx.y * gridDim.x + blockIdx.x,
           threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
               threadIdx.x,
           n);
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