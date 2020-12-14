#include "kernels.h"
#include <iostream>
#include <vector>
#include <chrono>

double time_ms(){
  auto t = std::chrono::high_resolution_clock::now();
  return t.time_since_epoch().count() / 1e6;
}

double averageTime(std::vector<double>& ts, int warmup){
  double t = 0;
  for (int i = warmup; i < ts.size(); i++) {
    t += ts[i];
  }
  return t / (ts.size() - warmup);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("need a input count\n");
    return -1;
  }
  size_t count = std::stoull(argv[1]);
  size_t buffer_size = count * sizeof(float);

  // allocate two device memory
  void* gpu_buffer1;
  void* gpu_buffer2;
  cudaMalloc(&gpu_buffer1, buffer_size);
  cudaMalloc(&gpu_buffer2, buffer_size);

  void* host_mapped_buffer;
  cudaHostAlloc(&host_mapped_buffer, buffer_size, cudaHostAllocMapped);

  std::vector<double> time_costs_gpu;
  cudaStream_t stream;
  StreamCreate(&stream);
  int warmup = 20;
  int repeat = 100;
  for (int i = 0; i < repeat; i++) {
    double start_time = time_ms();
    sumTwoBufferToFirst(gpu_buffer1, gpu_buffer2, count, stream);
    cudaStreamSynchronize(stream);
    time_costs_gpu.push_back(time_ms() - start_time);
  }

  printf("summation of %zu elements with two buffers on gpu, average time %f (ms)\n", count, averageTime(time_costs_gpu, warmup));

  std::vector<double> time_costs_mapped;
  for (int i = 0; i < repeat; ++i) {
    double start_time = time_ms();
    sumTwoBufferToFirst(gpu_buffer1, host_mapped_buffer, count, stream);
    cudaStreamSynchronize(stream);
    time_costs_mapped.push_back(time_ms() - start_time);
  }
  printf("summation of %zu elements with one buffer on cpu, average time %f (ms)\n", count, averageTime(time_costs_mapped, warmup));

  return 0;
}