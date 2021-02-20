#include <cstdlib>
#include <cstring>
#include "kernels.h"
#include "utils.h"


void dev2HostBandwidth(void* dev_buff, size_t nbytes, int repeat) {
  int warm_up = 5;
  void* pinned_host_mem;
  CUDACHECK(cudaHostAlloc(&pinned_host_mem, N_HOST_MEM_SLOTS * MEM_SLOT_SIZE,
                          cudaHostAllocMapped));
  double acc_time = 0;
  double acc_kernel_time = 0;
  cudaEvent_t start, stop;
  cudaStream_t stream;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int i = 0; i < repeat + warm_up; ++i) {
    hostDevShmInfo* task_info;
    hostAlloc<hostDevShmInfo>(&task_info, 1);
    task_info->head = N_HOST_MEM_SLOTS;
    task_info->tail = 0;
    task_info->size_idx = 0;
    for (int j = 0; j < N_HOST_MEM_SLOTS; ++j) {
      task_info->ptr_fifo[j] = (char*)pinned_host_mem + j * MEM_SLOT_SIZE;
      task_info->size_fifo[j] = 0;
    }
    void* kernel_args[3] = {&dev_buff, &task_info, &nbytes};
    double start_time = timeMs();
    CUDACHECK(cudaEventRecord(start, stream));
    CUDACHECK(cudaLaunchKernel((void*)netSendKernel, dim3(1), dim3(320), kernel_args, 0, stream));
    CUDACHECK(cudaEventRecord(stop, stream));
    size_t offset = 0;
    while (offset < nbytes) {
      if (task_info->head < task_info->tail + N_HOST_MEM_SLOTS) {
        // item to consume
        int _idx = task_info->size_idx;
        int real_size = task_info->size_fifo[_idx];
        // consume the buffer immediately
        task_info->head++;
        offset += real_size;
        task_info->size_idx = (_idx + 1) % N_HOST_MEM_SLOTS;
      }
    }
    cudaEventSynchronize(stop);

    if (i >= warm_up) {
      acc_time += (timeMs() - start_time);
      float kernel_time;
      cudaEventElapsedTime(&kernel_time, start, stop);
      acc_kernel_time += kernel_time;
    }
    // printf("time %f ms \n", timeMs() - start_time);
  }
  double avg_time = acc_time / repeat;
  double avg_kernel_time = acc_kernel_time / repeat;
  double bw = nbytes * 8 / avg_time / 1e6; // Gbps
  double kernel_bw = nbytes * 8 / avg_kernel_time / 1e6;
  printf("avg cost time %f ms, bandwidth %f Gbps, kernel bw %f Gbps\n", avg_time, bw, kernel_bw);
}

// TODO: test the send kernel that move the data from dev to host
// at host, launch a thread always move the head pointer ahead, 
// to pretend the memory buffer has been consumed
// measure the bandwidth
// -> expect to see 90Gbps
void testSendKernel() {
  int nelem = 8 * 1024 * 1024 + 5000; 
  size_t nbytes = nelem * sizeof(float);
  // create a sentinel buffer at host
  void* host_buff = malloc(nbytes);
  fillVals<float>((float*)host_buff, nelem);
  // copy buffer to dev
  void* dev_buff;
  CUDACHECK(cudaMalloc(&dev_buff, nbytes));
  CUDACHECK(cudaMemcpy(dev_buff, host_buff, nbytes, cudaMemcpyDefault));

  // alloc shared memory
  hostDevShmInfo* task_info;
  void* pinned_host_mem;
  hostAlloc<hostDevShmInfo>(&task_info, 1);
  task_info->tail = 0;
  task_info->head = N_HOST_MEM_SLOTS;
  CUDACHECK(cudaHostAlloc(&pinned_host_mem, N_HOST_MEM_SLOTS * MEM_SLOT_SIZE,
                          cudaHostAllocMapped));
  for (int i = 0; i < N_HOST_MEM_SLOTS; ++i) {
    task_info->ptr_fifo[i] = (char*)pinned_host_mem + i * MEM_SLOT_SIZE;
    task_info->size_fifo[i] = 0;
  }
  // launch netSendKernel to move the data from dev to host
  // first round do the data integrity check
  void* kernel_args[3] = {&dev_buff, &task_info, &nbytes};
  printf("kernel_args: %p, %p, %lu\n", dev_buff, (void*)task_info, nbytes);
  CUDACHECK(cudaLaunchKernel((void*)netSendKernel, dim3(1), dim3(32), kernel_args, 0, NULL));
  void* host_saveto = malloc(nbytes);
  size_t offset = 0;
  // printf("->>> info head %zu\n", task_info->head);
  while (offset < nbytes) {
    if (task_info->head < task_info->tail + N_HOST_MEM_SLOTS) {
      // item to consume
      int _idx = task_info->size_idx;
      int real_size = task_info->size_fifo[_idx];
      memcpy((char*)host_saveto + offset, task_info->ptr_fifo[_idx],
             real_size);
      task_info->head++;
      offset += real_size;
      task_info->size_fifo[_idx] = 0;
      task_info->size_idx = (_idx+1) % N_HOST_MEM_SLOTS;
      // printf("consumed size %d, head %zu, tail %zu, size_idx %d\n",  real_size, task_info->head, task_info->tail, task_info->size_idx);
    }
  }
  int match = memcmp(host_buff, host_saveto, nbytes);
  CUDACHECK(cudaDeviceSynchronize());
  printf("data integrity %s \n", match == 0? "true":"false");
  // then do bandwidth test
  dev2HostBandwidth(dev_buff, nbytes, 20);
}

int main() {
  printf("test kernels\n");
  testSendKernel();
}