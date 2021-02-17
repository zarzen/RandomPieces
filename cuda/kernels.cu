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

__global__ void dataMove(const volatile float* src, volatile float* dst, size_t count) {
  int n_threads = blockDim.x;
  int steps = count / n_threads;
  int remain = count - (steps * n_threads);
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int offset_idx = tid * steps;
  if (tid == n_threads - 1) {
    steps += remain;
  }
  int c = 0;
  while (c < steps) {
    dst[offset_idx + c] = src[offset_idx + c];
    ++c;
  }
}

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

inline __device__ void directStore128(Pack128* p, const Pack128* v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v->x), "l"(v->y) : "memory");
}

// inline __device__ void Store256(Pack256* p, const Pack256* v) {
//   asm volatile("st.volatile.global.v4.u64 [%0], {%1,%2,%3,%4};" :: "l"(p), "l"(v->x), "l"(v->y), "l"(v->w), "l"(v->z): "memory");
// }

__global__ void pack128Move(const Pack128* src, Pack128* dst, int count) {
  // printf("pack128Move, src %p, dst %p\n", (void*)src, (void*)dst);
  int nthreads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  int inc = nw * UNROLL * WARP_SIZE;
  int offset = w * UNROLL * WARP_SIZE + t;
  // printf("tid %d, w %d, nw %d, t %d, inc %d, offset %d, count %d\n", tid, w, nw, t, inc, offset, count);

  src = src+offset;
  dst += offset;

  while (offset < count) {
    // printf("tid %d, offset %d\n", tid, offset);
    Pack128 vals[UNROLL];

    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {Fetch128(vals[u], src+u*WARP_SIZE);} // locality, each wrap operates on consecutive datas

    #pragma unroll 
    for (int u = 0; u < UNROLL; ++u) {Store128(dst+u*WARP_SIZE, vals[u]);}

    src += inc;
    dst += inc;
    offset += inc;
    
  }

}

__global__ void pack128MoveUnroll1(const Pack128* src, Pack128* dst, int count) {
  // printf("pack128Move, src %p, dst %p\n", (void*)src, (void*)dst);
  int nthreads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  int inc = nw * WARP_SIZE;
  int offset = w * WARP_SIZE + t;
  // printf("tid %d, w %d, nw %d, t %d, inc %d, offset %d, count %d\n", tid, w, nw, t, inc, offset, count);

  src = src+offset;
  dst += offset;

  while (offset < count) {
    directStore128(dst, src);

    src += inc;
    dst += inc;
    offset += inc;
  }

}

// 7_kernels.compute_70.ptx, line 537; error   : Vector type too large, exceeds 128 bit limit
// __global__ void pack256Move(const Pack256* src, Pack256* dst, size_t count) {
//   int n_threads = blockDim.x;
//   int steps = count / n_threads;
//   int remain = count - (steps * n_threads);
//   int tid = blockDim.x * blockIdx.x + threadIdx.x;
//   int offset_idx = tid * steps;
//   if (tid == n_threads - 1) {
//     steps += remain;
//   }
//   int c = 0;
//   while (c < steps) {
//     // Store128(dst+offset_idx+c, src+offset_idx+c);
//     Store256(dst+offset_idx+c, src+offset_idx+c);
//     ++c;
//   }
// }

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