#include "kernels.h"
#include <device_launch_parameters.h>


inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

inline __device__ void directStore128(Pack128* p, Pack128* v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v->x), "l"(v->y) : "memory");
}

template<typename T>
inline __device__ void directVStore(T* d, T* s) {
  *d = *s;
}

template<int UNROLL>
__device__ __forceinline__ void unrollCopy128(Pack128* dst,
                                        Pack128* src,
                                        int count,
                                        int tid,
                                        int nw,
                                        int w,
                                        int t) {
  int offset = w * WARP_SIZE * UNROLL + t;
  int inc = nw * UNROLL * WARP_SIZE;

  dst += offset;
  src += offset;

  while (offset < count) {
    Pack128 vals[UNROLL];

    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      Fetch128(vals[u], src + u * WARP_SIZE);
    }

    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      Store128(dst + u * WARP_SIZE, vals[u]);
    }

    src += inc;
    dst += inc;
    offset += inc;
  }
}

__device__ __forceinline__ void
move128(Pack128* dst, Pack128* src, int& nelem, int& tid, int& nw, int& w, int& t) {
  int n_unroll_pack =
      (nelem / (WARP_SIZE * DEFAULT_UNROLL)) * (WARP_SIZE * DEFAULT_UNROLL);
  // int n_unroll_bytes = n_unroll_pack * sizeof(Pack128);
  int n_pack_remain = nelem - n_unroll_pack;

  // use unroll version first
  unrollCopy128<DEFAULT_UNROLL>(dst, src, n_unroll_pack, tid, nw, w, t);

  unrollCopy128<1>(dst+n_unroll_pack, src + n_unroll_pack, n_pack_remain, tid, nw, w, t);

}

inline __device__ void copyChars(char* dst, char* src, int count, int tid, int nw, int w, int t) {
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
    // make the write to shared memory visible to host threads
    __threadfence_system();
  }
  size_idx = (++size_idx) % N_HOST_MEM_SLOTS;
  // only 1 block, so sync within threads is good enough
  // besides, sync at this point makes copy kernel behave correctly
  // TODO: figure out why
  __syncthreads(); 

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
    move128((Pack128*)(ptr_fifo[size_idx]), (Pack128*)(src + send_offset),
            n_pack128_each_slot, tid, nw, w, t);
    
    postSend(tail, size_idx, size_fifo, host_slot_size, tid);
    send_offset += host_slot_size;
  }

  int last_chunk = count_bytes - n_steps * host_slot_size;
  int remain = last_chunk;

  if (remain > 0) {
    waitSend(head, tail);
    int n_pack128 = remain / sizeof(Pack128);
    move128((Pack128*)(ptr_fifo[size_idx]), (Pack128*)(src + send_offset),
            n_pack128, tid, nw, w, t);
    int copied_bytes = n_pack128 * sizeof(Pack128);
    send_offset += copied_bytes;
    remain = remain - copied_bytes;
    if (remain > 0) {
      void* dst = ptr_fifo[size_idx] + copied_bytes;
      copyChars((char*)dst, src + send_offset, remain, tid, nw, w, t);
    }
    postSend(tail, size_idx, size_fifo, last_chunk, tid);
  }

}

inline __device__ void waitRecv(volatile size_t* head, volatile size_t* tail) {
  while(*head - *tail >= N_HOST_MEM_SLOTS) {
    // nothing to consume
  }
  // __threadfence_system();
}

inline __device__ void postRecv(int& tid, volatile size_t* head, int& size_idx) {
  if (tid == 0) {
    ++(*head);
  }
  size_idx = (size_idx + 1) % N_HOST_MEM_SLOTS;
  // because only use one block
  __syncthreads();
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

    move128((Pack128*)(dst + offset), (Pack128*)ptr_fifo[size_idx],
            n_pack128_each_slot, tid, nw, w, t);

    postRecv(tid, head, size_idx);
    offset += host_slot_size;
  }

  int last_chunk = count_bytes - offset;
  int remain = last_chunk;

  if (remain > 0) {
    waitRecv(head, tail);
    int n_pack128 = remain / sizeof(Pack128);
    move128((Pack128*)(dst + offset), (Pack128*)ptr_fifo[size_idx], n_pack128,
            tid, nw, w, t);
    int copied_bytes = n_pack128 * sizeof(Pack128);
    offset += copied_bytes;
    remain -= copied_bytes;

    if (remain > 0) {
      copyChars(dst + offset, ptr_fifo[size_idx] + copied_bytes, remain, tid,
                nw, w, t);
    }

    postRecv(tid, head, size_idx);

  }
}

__global__ void p2pSendKernel(void* dst_buff, void* src_buff, size_t count) {
  int nthreads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  Pack128* pack128_dst = (Pack128*)dst_buff;
  Pack128* pack128_src = (Pack128*)src_buff;

  int pack128_count = count / sizeof(Pack128);
  move128(pack128_dst, pack128_src, pack128_count, tid, nw, w, t);

  size_t pack128_offset = pack128_count * sizeof(Pack128);
  size_t remain = count - pack128_offset;
  if (remain > 0) {
    copyChars((char*)dst_buff + pack128_offset,
              (char*)src_buff + pack128_offset, remain, tid, nw, w, t);
  }
}