#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <sstream>

#define WARP_SIZE 32
#define SLOT_SIZE 16 // in bytes
#define N_SLOT 4

typedef ulong2 Pack128;

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

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

inline __device__ void directStore128(Pack128* p, const Pack128* v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v->x), "l"(v->y) : "memory");
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
    // printf("tid %d, from ptr %p, to ptr %p\n", tid, (void*)src, (void*)dst);
    // printf("tid %d, nw %d, w %d, t %d, offset %d\n", tid, nw, w, t, offset);
    // Pack128 v;
    // Fetch128(v, src);
    // Store128(dst, v);

    // use directStore cause incorrectness: fetches wrong data from pinned memory
    directStore128(dst, src); 

    src += inc;
    dst += inc;
    offset += inc;
    
  }
}

struct CtrlBlock {
  size_t head;
  size_t tail;
  void* buffers[N_SLOT];
};

inline __device__ void wait(volatile size_t* head, volatile size_t* tail) {
  while(*head - *tail >= N_SLOT) {
    // nothing to consume: if there are some slots occupied the head - tail < N_SLOT
  }
}

inline __device__ void barrier(int& nthreads) {
  asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
}

__global__ void streamCopy(void* dev_buff, CtrlBlock* ctrl, size_t nbytes) {
  int nthreads = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int w = tid / WARP_SIZE;       // Warp number
  int nw = nthreads / WARP_SIZE; // Number of warps
  int t = tid % WARP_SIZE;       // Thread (inside the warp)

  // printf("tid %d, dev ptr %p\n", tid, dev_buff);

  volatile size_t* head = &(ctrl->head);
  volatile size_t* tail = &(ctrl->tail);

  int slot_idx = 0;
  int n_per_slot = SLOT_SIZE / sizeof(Pack128);

  size_t offset = 0;
  char* copy_to = (char*)dev_buff;
  int n_steps = nbytes / SLOT_SIZE;

  for (int i = 0; i < n_steps; ++i) {
    wait(head, tail);
    void* src_ptr = ctrl->buffers[slot_idx];
    // if (tid == 0) {
    //   printf("tid %d, step %d, slot_idx %d, from ptr %p, to ptr %p, head %lu, tail %lu\n", tid, i,
    //          slot_idx, src_ptr, (void*)(copy_to + offset), *head, *tail);
    // }

    copy128((Pack128*)(copy_to + offset), (Pack128*)src_ptr, n_per_slot, 
              tid, nw, w, t);
    barrier(nthreads);

    if (tid == 0) {
      (*head)++;
    }
    slot_idx = (slot_idx + 1) % N_SLOT;

    offset += SLOT_SIZE;
  }
}

static inline void fillVals(float* buff, size_t count) {
  for (int i = 0; i < count; ++i) {
    float e = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    buff[i] = e;
  }
}

void printFloats(const char* prefix, float* buffer, int n) {
  std::stringstream ss;
  for (int i = 0; i < n ; ++i) {
    ss << *(buffer+i);
    if (i != n -1 ) ss << ",";
  }
  printf("%s:%s\n", prefix, ss.str().c_str());
}

int main() {
  int nelem = 32; // 32 floats
  size_t nbytes = nelem * sizeof(float);

  // init memory buffers for control block
  CtrlBlock* ctrl_block;
  CUDACHECK(cudaHostAlloc(&ctrl_block, sizeof(CtrlBlock), cudaHostAllocMapped));
  ctrl_block->tail = 0;
  ctrl_block->head = N_SLOT;

  for (int i = 0; i < N_SLOT; ++i) {
    void* data_buff;
    CUDACHECK(cudaHostAlloc(&data_buff, SLOT_SIZE, cudaHostAllocMapped));
    ctrl_block->buffers[i] = data_buff;
  }

  // init memory buffer for experiment
  void *host_data_buff, *host_tmp_buff, *dev_buff;
  host_data_buff = malloc(nbytes);
  host_tmp_buff = malloc(nbytes);
  fillVals((float*)host_data_buff, nelem);
  CUDACHECK(cudaMalloc(&dev_buff, nbytes));

  // launch stream copy kernel
  void* kernel_args[3] = {&dev_buff, &ctrl_block, &nbytes};
  printf("dev ptr %p\n", dev_buff);
  CUDACHECK(cudaLaunchKernel((void*)streamCopy, dim3(1), dim3(32), kernel_args, 0, NULL));

  // launch our control log at host
  size_t offset = 0;
  int slot_idx = 0;
  while (offset < nbytes) {
    if (ctrl_block->head > ctrl_block->tail) {
      // there is memory block to store data
      int chunk_size = SLOT_SIZE;
      if (nbytes - offset < SLOT_SIZE) {
        // fill the full memory buff
        chunk_size = nbytes - offset;
      }

      memcpy(ctrl_block->buffers[slot_idx], (char*)host_data_buff + offset,
             chunk_size);
      printf("buffer ptr %p\n", ctrl_block->buffers[slot_idx]);
      printFloats("copied to pinned buffer", (float*)ctrl_block->buffers[slot_idx], 
                  SLOT_SIZE / sizeof(float));
      slot_idx = (slot_idx + 1) % N_SLOT;
      ctrl_block->tail++;
      offset += chunk_size;
    }
  }

  CUDACHECK(cudaDeviceSynchronize());

  // copy data to host_tmp buffer for verify
  CUDACHECK(cudaMemcpy(host_tmp_buff, dev_buff, nbytes, cudaMemcpyDefault));
  int match = memcmp(host_data_buff, host_tmp_buff, nbytes);

  if (match != 0) {
    printf("device buffer != original data buffer\n");
    printFloats("reference buffer", (float*)host_data_buff, nelem);
    printFloats("device buffer", (float*)host_tmp_buff, nelem);
  } else {
    printf("device buffer matches original data buffer on host\n");
  }
}