#include "kernels.h"
#include <device_launch_parameters.h>


inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

inline __device__ void directStore128(Pack128* p, const Pack128* v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v->x), "l"(v->y) : "memory");
}

template<typename T>
inline __device__ void directVStore(T* d, T* s) {
  *d = *s;
}

__device__ __forceinline__ void copy128(Pack128* dst,
                                        Pack128* src,
                                        size_t count,
                                        int tid,
                                        int nw,
                                        int w,
                                        int t) {
  int offset = w  * WARP_SIZE + t;
  int inc = nw * WARP_SIZE;
  dst += offset;
  src += offset;
  while(offset < count) {
    directStore128(dst, src);

    src += inc;
    dst += inc;
    offset += inc;
  }
}

inline __device__ void copyChars(char* dst, char* src, size_t count, int tid, int nw, int w, int t) {
  int offset = w * WARP_SIZE + t;
  int inc = nw * WARP_SIZE;
  src += offset;
  dst += offset;

  while(offset < count) {
    *dst = *src;

    src += inc;
    dst += inc;
    offset += inc;
  }
}

inline __device__ void waitSend(volatile size_t* head, volatile size_t* tail) {
  size_t cached_head = *head;
  while (cached_head <= *tail) { // head is controlled by the consumer
    cached_head = *head;
  }
}

inline __device__ void postSend(volatile size_t* tail,
                                int& size_idx,
                                volatile int* size_fifo,
                                int& n_bytes,
                                int tid) {
  if (tid == 0) {
    size_fifo[size_idx] = n_bytes;
    (*tail)++;
  }

  size_idx = (++size_idx) % N_HOST_MEM_SLOTS;
}

inline __device__ void barrier(int& nthreads) {
  asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
}

__global__ void netSendKernel(void* send_buff, struct hostDevShmInfo* info, size_t count_bytes) {
  int nthreads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  volatile size_t* head = &(info->head);
  volatile size_t* tail = &(info->tail);
  volatile int* size_fifo = info->size_fifo;
  int size_idx = 0;
  char* ptr_fifo[N_HOST_MEM_SLOTS];

  // store the ptrs locally, not fetch from global memory
  #pragma unroll
  for (int i = 0 ; i < N_HOST_MEM_SLOTS; ++i) {
    ptr_fifo[i] = (char*)info->ptr_fifo[i];
  }

  int host_slot_size = MEM_SLOT_SIZE;
  size_t n_steps = count_bytes / host_slot_size;
  int n_pack128_each_slot = host_slot_size / sizeof(Pack128);
  size_t send_offset = 0;
  char* src = (char*)send_buff;

  // always fill the Mem slot
  for (int i = 0; i < n_steps; ++i) {
    waitSend(head, tail);
    // copy to host memory
    copy128((Pack128*)(ptr_fifo[size_idx]), (Pack128*)(src + send_offset),
            n_pack128_each_slot, tid, nw, w, t);
    barrier(nthreads);
    postSend(tail, size_idx, size_fifo, host_slot_size, tid);
    send_offset += host_slot_size;
  }

  int last_chunk = count_bytes - n_steps * host_slot_size;
  int remain = last_chunk;

  if (remain > 0) {
    waitSend(head, tail);
    int n_pack128 = remain / sizeof(Pack128);
    copy128((Pack128*)(ptr_fifo[size_idx]), (Pack128*)(src + send_offset),
            n_pack128, tid, nw, w, t);
    int copied_bytes = n_pack128 * sizeof(Pack128);
    send_offset += copied_bytes;
    remain = remain - copied_bytes;
    if (remain > 0) {
      void* dst = ptr_fifo[size_idx] + copied_bytes;
      copyChars((char*)dst, src + send_offset, remain, tid, nw, w, t);
    }
    barrier(nthreads);
    postSend(tail, size_idx, size_fifo, last_chunk, tid);
  }

}

inline __device__ void waitRecv(volatile size_t* head, volatile size_t* tail) {
  while(*head - *tail >= N_HOST_MEM_SLOTS) {
    // nothing to consume
  }
}

inline __device__ void postRecv(int& tid, volatile size_t* head, int& size_idx) {
  if (tid == 0) {
    ++(*head);
  }
  size_idx = (size_idx + 1) % N_HOST_MEM_SLOTS;
}

__global__ void netRecvKernel(void* recv_buff, struct hostDevShmInfo* info, size_t count_bytes) {
  int nthreads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  volatile size_t* head = &(info->head);
  volatile size_t* tail = &(info->tail);
  // volatile int* size_fifo = info->size_fifo;
  int size_idx = 0;
  char* ptr_fifo[N_HOST_MEM_SLOTS];

  // store the ptrs locally, not fetch from global memory
  #pragma unroll
  for (int i = 0 ; i < N_HOST_MEM_SLOTS; ++i) {
    ptr_fifo[i] = (char*)info->ptr_fifo[i];
  }

  int host_slot_size = MEM_SLOT_SIZE;
  size_t n_steps = count_bytes / host_slot_size;
  int n_pack128_each_slot = host_slot_size / sizeof(Pack128);
  size_t offset = 0;
  char* dst = (char*)recv_buff;

  // always fill the Mem slot
  for (int i = 0; i < n_steps; ++i) {
    waitRecv(head, tail);
    // copy to device
    copy128((Pack128*)(dst + offset), (Pack128*)ptr_fifo[size_idx],
            n_pack128_each_slot, tid, nw, w, t);
    barrier(nthreads);
    postRecv(tid, head, size_idx);
    offset += host_slot_size;
    // printf("tid %d, step %d, offset %lu, head %lu, tail %lu, size_idx %d \n", tid, i, offset, *head, *tail, size_idx);
  }

  int last_chunk = count_bytes - offset;
  int remain = last_chunk;

  if (remain > 0) {
    waitRecv(head, tail);
    int n_pack128 = remain / sizeof(Pack128);
    copy128((Pack128*)(dst + offset), (Pack128*)ptr_fifo[size_idx], n_pack128,
            tid, nw, w, t);
    int copied_bytes = n_pack128 * sizeof(Pack128);
    offset += copied_bytes;
    remain -= copied_bytes;

    if (remain > 0) {
      copyChars(dst + offset, ptr_fifo[size_idx] + copied_bytes, remain, tid,
                nw, w, t);
    }
    barrier(nthreads);
    postRecv(tid, head, size_idx);
  }
}