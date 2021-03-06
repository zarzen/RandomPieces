#include <stdio.h>
#include "cuda_runtime.h"
#include "common.h"
#include <unistd.h>
#include <stdint.h>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <numeric>
#include <string>
#include <thread>

using std::vector;
int nDevices = 4;


static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

void initNccl(int* argc, char*** argv, ncclComm_t& comm, int& myRank, int& nRanks, int& local_rank) {
  
  //initializing MPI
  MPICHECK(MPI_Init(argc, argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  //get NCCL unique ID at rank 0 and broadcast it to all others
  ncclUniqueId id;
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  local_rank = myRank % nDevices;
  printf("local_rank %d\n", local_rank);
  cudaSetDevice(local_rank);
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  printf("[MPI Rank %d] Success \n", myRank);
}

void whileThread(bool* shutdown) {
  int a = 0;
  int b = 10;
  while (!(*shutdown)) {
    a = (a + 1) % b;
  }
}

void launchBackgroundThreads(std::vector<std::thread>& threads, int n, bool* shutdown) {
  for (int i = 0; i < n; ++i) {
    threads.push_back(std::thread(whileThread, shutdown));
  }
}

void exitBackgroundThreads(std::vector<std::thread>& threads, bool* shutdown) {
  *shutdown = true;
  for (auto& t: threads) {
    t.join();
  }
}

int main(int argc, char* argv[])
{
  bool shutdown=false;
  std::vector<std::thread> back_thds;
  launchBackgroundThreads(back_thds, 6, &shutdown);

  if (argc > 1) {
    nDevices = std::stoi(argv[1]);
  }
  int myRank, nRanks, local_rank;
  int next_peer, pre_peer;
  ncclComm_t comm;
  initNccl(&argc, &argv, comm, myRank, nRanks, local_rank);
  next_peer = (myRank + 1) % nRanks;
  pre_peer = (myRank+nRanks - 1) % nRanks;
  printf("rank %d, local rank %d, send to %d, receive from %d \n", myRank, local_rank, next_peer, pre_peer);

  int buffer_size = 256 *  1024 * 1024;
  float *sendbuff, *recvbuff, *result, *data_val;
  cudaStream_t stream;
  
  data_val = (float*)malloc(buffer_size);
  result = (float*)malloc(buffer_size);

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaMalloc(&sendbuff, buffer_size));
  CUDACHECK(cudaMalloc(&recvbuff, buffer_size));
  //assign value to sendbuff
  CUDACHECK(cudaMemset(recvbuff, 0, buffer_size));
  CUDACHECK(cudaMemset(sendbuff, 0, buffer_size));
  CUDACHECK(cudaStreamCreate(&stream));

  int n_repeat = 100;

  std::vector<cudaEvent_t> start_events;
  std::vector<cudaEvent_t> stop_events;
  for (int i  = 0; i < n_repeat; ++i) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    start_events.push_back(start);
    stop_events.push_back(stop);
  }
  int warm_up = 5;

  for (int i = 0; i < 7; ++i) {
    size_t trans_bytes = (size_t) pow(2, i) * 1024 * 1024;
    size_t count = trans_bytes / sizeof(float);
    vector<float> time_costs;

    for (int j = 0; j < n_repeat; ++j) {
      cudaEventRecord(start_events[j], stream);
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclSend(sendbuff, count, ncclFloat, next_peer, comm, stream));
      NCCLCHECK(ncclRecv(recvbuff, count, ncclFloat, pre_peer, comm, stream));
      NCCLCHECK(ncclGroupEnd());
      cudaEventRecord(stop_events[j], stream);
    }

    cudaStreamSynchronize(stream);
    for (int j=0; j < n_repeat; ++j) {
      //completing NCCL operation by synchronizing on the CUDA stream
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start_events[j], stop_events[j]);
      if (j >= warm_up) 
        time_costs.push_back(milliseconds);
    }
    
    double avg = std::accumulate(time_costs.begin(), time_costs.end(), 0.0) / time_costs.size();
    printf("buffer size %zu, send&recv cost %f ms, bw %f Gbps\n", trans_bytes, avg, trans_bytes * 8 / avg / 1e6);
  }

  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  free(data_val);
  free(result);

  //finalizing NCCL
  ncclCommDestroy(comm);

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  exitBackgroundThreads(back_thds, &shutdown);
  return 0;
}