#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <numeric>
#include <string>

using std::vector;
int nDevices = 4;

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t rank = cmd;                             \
  if (rank!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(rank));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


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


int main(int argc, char* argv[])
{
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
  cudaStream_t s;
  
  data_val = (float*)malloc(buffer_size);
  result = (float*)malloc(buffer_size);

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaMalloc(&sendbuff, buffer_size));
  CUDACHECK(cudaMalloc(&recvbuff, buffer_size));
  //assign value to sendbuff
  CUDACHECK(cudaMemset(recvbuff, 0, buffer_size));
  CUDACHECK(cudaMemset(sendbuff, 0, buffer_size));
  CUDACHECK(cudaStreamCreate(&s));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 9; ++i) {
    size_t trans_bytes = (size_t) pow(2, i) * 1024 * 1024;
    size_t count = trans_bytes / sizeof(float);
    vector<float> time_costs;

    for (int j = 0; j < 50; ++j) {
      cudaEventRecord(start);
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclSend(sendbuff, count, ncclFloat, next_peer, comm, s));
      NCCLCHECK(ncclRecv(recvbuff, count, ncclFloat, pre_peer, comm, s));
      NCCLCHECK(ncclGroupEnd());
      cudaEventRecord(stop);
      //completing NCCL operation by synchronizing on the CUDA stream
      CUDACHECK(cudaStreamSynchronize(s));
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      time_costs.push_back(milliseconds);
    }
    double avg = std::accumulate(time_costs.begin(), time_costs.end(), 0.0) / time_costs.size();
    printf("buffer size %zu, send&recv cost %f ms\n", trans_bytes, avg);
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

  return 0;
}