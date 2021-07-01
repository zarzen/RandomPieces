#include "common.h"
#include "cuda_runtime.h"
#include <string>
#include <iostream>
#include <vector>

int rank;
int nranks;
int local_rank;
int local_size;
int device_idx;
int nsub_groups;

ncclComm_t world_comm; // for all ranks
ncclComm_t group_comm; // only for local_rank=0
ncclComm_t local_comm; // inside each group
void init_nccl_communicators() {
  int ngroups = nsub_groups + 2;
  std::vector<ncclUniqueId> ids;
  if (rank == 0) {
    for (int i = 0; i < ngroups; ++i) {
      ncclUniqueId id;
      ncclGetUniqueId(&id);
      ids.push_back(id);
    }
  }

  MPICHECK(MPI_Bcast(ids.data(), sizeof(ncclUniqueId) * ngroups, MPI_BYTE, 0,
                     MPI_COMM_WORLD));

  cudaSetDevice(device_idx);

  // use first id to init world_comm
  NCCLCHECK(ncclCommInitRank(&world_comm, nranks, ids[0], rank));
  printf("Init global communicator, size %d, rank %d\n", nranks, rank);

  int group_idx = rank / nsub_groups;
  NCCLCHECK(ncclCommInitRank(&group_comm, nsub_groups, ids[1], group_idx));
  printf("Init group communicator\n");

  ncclUniqueId local_group_id = ids[group_idx+2];
  NCCLCHECK(
      ncclCommInitRank(&local_comm, local_size, local_group_id, local_rank));
  printf(
      "Init local communicator, local_size %d, local_rank %d\n, global_rank %d",
      local_size, local_rank, rank);
}

void* gpu_buffer;
size_t nelem;
size_t buffer_size; // init in main
void init_buffers() {
  cudaSetDevice(local_rank);
  CUDACHECK(cudaMalloc(&gpu_buffer, buffer_size));
}

void allreduce_global(int warm_up=5, int repeat=10) {
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < warm_up; i++) {
    NCCLCHECK(ncclAllReduce(gpu_buffer, gpu_buffer, nelem, ncclFloat32, ncclSum,
                            world_comm, stream));
  }

  float avg_time = 0;
  for (int i = 0; i < repeat; ++i) {
    cudaEventRecord(start, stream);
    NCCLCHECK(ncclAllReduce(gpu_buffer, gpu_buffer, nelem, ncclFloat32, ncclSum,
                            world_comm, stream));
    cudaEventRecord(stop, stream);

    cudaStreamSynchronize(stream);
    float _cost= 0;
    cudaEventElapsedTime(&_cost, start, stop);
    avg_time += (_cost / float(repeat));
  }
  float ratio = 2 * (float(nranks) - 1) / float(nranks);
  float bw = ratio * buffer_size / 1e9 / (avg_time * 1e-3);
  printf("ring allreduce on global buffer size %lu, bw %f GB/s \n", buffer_size, bw);
}

// reduce in local groups 
// allreduce among groups
// bcast in local groups
void allreduce_hierarchy(int warm_up=5, int repeat=10) {
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float avg_time = 0;
  for (int i = 0; i < warm_up + repeat; ++i) {
    cudaEventRecord(start, stream);
    // local reduce to local_rank = 0
    NCCLCHECK(ncclReduce(gpu_buffer, gpu_buffer, nelem, ncclFloat32, ncclSum, 0,
                         local_comm, stream));
    if (local_rank == 0) {
      // all reduce among groups
      NCCLCHECK(ncclAllReduce(gpu_buffer, gpu_buffer, nelem, ncclFloat32,
                              ncclSum, group_comm, stream));
    }

    // bcast back to local
    NCCLCHECK(ncclBroadcast(gpu_buffer, gpu_buffer, nelem, ncclFloat32, 0,
                            local_comm, stream));
    cudaEventRecord(stop, stream);

    cudaStreamSynchronize(stream);
    float _cost= 0;
    cudaEventElapsedTime(&_cost, start, stop);
    if (i >= warm_up) {
      avg_time += (_cost / float(repeat));
    }
  }

  float ratio = 2 * (float(nsub_groups) - 1) / float(nsub_groups);
  float bw = ratio * buffer_size / 1e9 / (avg_time * 1e-3);
  printf("ring allreduce on global buffer size %lu, bw %f GB/s \n", buffer_size, bw);

}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "input the size of local, and nelem of floats";
  }
  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  // setup vars
  local_size = std::stoi(argv[1]);
  device_idx = rank % local_size;
  local_rank = device_idx;
  nsub_groups = nranks / local_size;

  nelem = std::stoi(argv[2]);
  buffer_size =  nelem * sizeof(float);

  init_nccl_communicators();
  init_buffers();

  allreduce_global();

  allreduce_hierarchy();
}