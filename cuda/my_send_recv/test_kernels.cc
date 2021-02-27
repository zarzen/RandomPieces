#include <cstdlib>
#include <cstring>
#include "kernels.h"
#include "utils.h"

void init_buffers(void** host_buff,
                  void** host_buff_datacheck,
                  void** dev_buff,
                  size_t& nbytes,
                  bool init_dev_buff = true) {
  *host_buff = malloc(nbytes);
  *host_buff_datacheck = malloc(nbytes);
  fillVals<float>((float*)*host_buff, nbytes/sizeof(float));
  CUDACHECK(cudaMalloc(dev_buff, nbytes));
  if (init_dev_buff)
    CUDACHECK(cudaMemcpy(*dev_buff, *host_buff, nbytes, cudaMemcpyDefault));
}

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

// 77 Gbps bandwidth
void testSendKernel(size_t& nelem) {
  size_t nbytes = nelem * sizeof(float);
  // create a sentinel buffer at host
  void* dev_buff;
  void* host_buff;
  void* host_check_buff;
  init_buffers(&host_buff, &host_check_buff, &dev_buff, nbytes);

  // alloc shared memory
  hostDevShmInfo* task_info;
  allocDevCtrl(&task_info);

  double buffer_sum = floatSummary((float*)host_buff, nelem);
  LOG_INFO("host buff summary %f", buffer_sum);

  // launch netSendKernel to move the data from dev to host
  // first round do the data integrity check
  void* kernel_args[3] = {&dev_buff, &task_info, &nbytes};
  printf("kernel_args: %p, %p, %lu\n", dev_buff, (void*)task_info, nbytes);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  CUDACHECK(cudaLaunchKernel((void*)netSendKernel, dim3(1), dim3(320), kernel_args, 0, stream));
  void* host_saveto = host_check_buff;
  int first_match = memcmp(host_buff, host_check_buff, nbytes);
  LOG_INFO("first match %s", first_match == 0? "true":"false");

  size_t offset = 0;
  double test_sum_ = 0;
  // printf("->>> info head %zu\n", task_info->head);
  while (offset < nbytes) {
    if (task_info->head < task_info->tail + N_HOST_MEM_SLOTS) {
      // item to consume
      int _idx = task_info->size_idx;
      int real_size = task_info->size_fifo[_idx];

      memcpy((char*)host_saveto + offset, task_info->ptr_fifo[_idx],
             real_size);
      // std::this_thread::sleep_for(std::chrono::milliseconds(5));
      test_sum_ += floatSummary((float*)task_info->ptr_fifo[_idx], real_size / sizeof(float));

      task_info->head++;
      offset += real_size;
      task_info->size_fifo[_idx] = 0;
      task_info->size_idx = (_idx+1) % N_HOST_MEM_SLOTS;
      // printf("consumed size %d, head %zu, tail %zu, size_idx %d\n",  real_size, task_info->head, task_info->tail, task_info->size_idx);
    }
  }
  int match = memcmp(host_buff, host_saveto, nbytes);
  CUDACHECK(cudaDeviceSynchronize());
  LOG_INFO("saveto_buf sum %f", floatSummary((float*)host_saveto, nelem));
  printf("data integrity %s \n", match == 0? "true":"false");
  LOG_INFO("test_send_sum %f", test_sum_);
  // then do bandwidth test
  dev2HostBandwidth(dev_buff, nbytes, 20);
}

void testRecvKernel(size_t nelem){
  size_t nbytes = nelem * sizeof(float);
  void *host_buff, *host_check_buff, *dev_buff;
  init_buffers(&host_buff, &host_check_buff, &dev_buff, nbytes, false);

  // alloc shared memory
  hostDevShmInfo* task_info;
  allocDevCtrl(&task_info);

  void* kernel_args[3] = {&dev_buff, &task_info, &nbytes};
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  int repeat = 10;
  int warm_up = 5;
  for (int i = 0; i < repeat; ++i) {
    double start_time = timeMs();
    CUDACHECK(cudaLaunchKernel((void*)netRecvKernel, dim3(1), dim3(320), kernel_args, 0, stream));
    size_t offset = 0;
    int slot_idx = 0;
    while (offset < nbytes) {
      if (task_info->head > task_info->tail) {
        // there is memory slots
        int chunk_size = MEM_SLOT_SIZE;
        if (nbytes - offset < MEM_SLOT_SIZE) {
          // fill the full memory buff
          chunk_size = nbytes - offset;
        }
        double start_time = timeMs();

        // std::this_thread::sleep_for(std::chrono::milliseconds(3));
        memcpy(task_info->ptr_fifo[slot_idx], (char*)host_buff + offset,
               chunk_size);

        // LOG_INFO("head %lu, tail %lu, slot_idx %d, ptr %p", task_info->head,
        //          task_info->tail, slot_idx, task_info->ptr_fifo[slot_idx]);
        // printFloats("intermediate", (float*)task_info->ptr_fifo[slot_idx],
        // 4);

        // printf("moved %d into slot%d, head %lu, tail %lu\n", chunk_size,
        // slot_idx, task_info->head, task_info->tail);
        task_info->size_fifo[slot_idx] = chunk_size;
        slot_idx = (slot_idx + 1) % N_HOST_MEM_SLOTS;
        offset += chunk_size;
        task_info->tail++;

        // printf("memcpy bw %f Gbps\n", chunk_size * 8 / (timeMs() -
        // start_time) / 1e6);
      }
    }
    CUDACHECK(cudaDeviceSynchronize());
    double end_time = timeMs();
    CUDACHECK(cudaMemcpy(host_check_buff, dev_buff, nbytes, cudaMemcpyDefault));
    int match = memcmp(host_buff, host_check_buff, nbytes);
    double bw = nbytes * 8 / (end_time - start_time) / 1e6;
    // printFloats("reference buffer", (float*)host_buff, 48);
    // printFloats("recv buffer", (float*)host_check_buff, 48);
    printf("recv kernel integrity check %s, bw %f Gbps \n",
           match == 0 ? "true" : "false", bw);
  }
  
}

int main() {
  size_t nelem = 4 * 1024 * 1024 + 1024; 
  // size_t nelem = 48;

  printf("test kernels\n");
  testSendKernel(nelem);
  testRecvKernel(nelem);
}