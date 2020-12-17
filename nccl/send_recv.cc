#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <cstdlib>

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


int main(int argc, char* argv[])
{
  int size = 4;
  int myRank, nRanks, localRank = 0;
  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }
  //printf("LocalRank = %d, TotalRank = %d\n", nRanks, localRank);

  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff, *result;
  float *data_val;
  cudaStream_t s;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  
  int total_bytes = size * sizeof(float);
  data_val = (float*)malloc(total_bytes);
  result = (float*)malloc(total_bytes);

  printf("Rank %d data (%d bytes) = [", total_bytes, localRank);
  for(int i = 0; i < size; ++i) {
      data_val[i] = myRank * size + i + 1;
      printf(" %f,", data_val[i]);
  }
  printf("]\n");

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  //assign value to sendbuff
  CUDACHECK(cudaMemcpy(sendbuff, data_val, size * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //communicating using NCCL (P2P)
  int slice = size / nRanks;
  size_t exchange_bytes = slice * sizeof(float);
  NCCLCHECK(ncclGroupStart());
  int offset = slice * myRank;
  int target = nRanks - myRank - 1;
  printf("rank %d send/recv %d bytes from offset %d to rank %d\n", myRank, exchange_bytes, offset, target);
  //rank0: send(offset_0, count_2, to_rank_1); recv(offset_0, count_2, from_rank_1);
  //rank1: send(offset_2, count_2, to_rank_0); recv(offset_2, count_2, from_rank_0);
  NCCLCHECK(ncclSend(sendbuff + offset, slice, ncclFloat, target, comm, s));
  NCCLCHECK(ncclRecv(sendbuff + offset, slice, ncclFloat, target, comm, s));
  NCCLCHECK(ncclGroupEnd());

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  //check result
  CUDACHECK(cudaMemcpy(result, sendbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
  printf("AllReduced result for rank %d = [", myRank);
  for(int i = 0; i < size; ++i){
    printf(" %f,", result[i]);
  }
  printf("]\n");

  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}