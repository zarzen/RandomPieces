#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>

double timeMs() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count() /
         1e6;
};

int main() {
  int n_slot = 4;
  int slot_size = 512 * 1024;
  int n_repeat = 10;

  void* host_buff;
  void* dev_buff;

  cudaHostAlloc(&host_buff, n_slot * slot_size, cudaHostAllocMapped);
  cudaMalloc(&dev_buff, n_slot * slot_size);

  std::vector<cudaEvent_t> sync_events;
  for (int i = 0; i < n_slot; ++i) {
    cudaEvent_t e;
    cudaEventCreate(&e);
    sync_events.push_back(e);
  }

  cudaError_t ret;
  cudaStream_t stream;
  ret = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  for (int r = 0; r < n_repeat; ++r) {
    double s = timeMs();

    for (int i = 0; i < n_slot; ++i) {
      ret = cudaMemcpyAsync((char*)dev_buff + i * slot_size,
                            (char*)host_buff + i * slot_size, slot_size,
                            cudaMemcpyDefault, stream);
      if (ret != cudaSuccess) {
        printf("%d: error while memcpyasync", __LINE__);
      }
      cudaEventRecord(sync_events[i], stream);
    }

    while (cudaEventQuery(sync_events[n_slot - 1]) != cudaSuccess) {
    }
    double e = timeMs();
    printf("copy to device progressively bw: %f Gbps\n", n_slot * slot_size * 8 / (e - s) / 1e6);
  }
}