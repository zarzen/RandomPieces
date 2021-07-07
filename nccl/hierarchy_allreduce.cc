#include "common.h"
#include "cuda_runtime.h"
#include <string>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cstdlib>

int rank;
int nranks;
int local_rank;
int local_size;
int device_idx;
int nsub_groups;

ncclComm_t world_comm; // for all ranks
ncclComm_t group_comm; // for ranks has same local_rank
ncclComm_t local_comm; // inside each group
void init_nccl_communicators() {
  int world_size = nranks;
  int my_rank = rank; // avoid compiler optimization 
  cudaSetDevice(device_idx);
  printf("nranks %d, rank %d, device %d \n", nranks, rank, device_idx);

  int n_unique_ids = nsub_groups + local_size + 1;
  std::vector<ncclUniqueId> ids;
  if (rank == 0) {
    for (int i = 0; i < n_unique_ids; ++i) {
      ncclUniqueId id;
      ncclGetUniqueId(&id);
      ids.push_back(id);
    }
  } else {
    for (int i = 0; i < n_unique_ids; ++i) {
      ncclUniqueId id; // place hold
      ids.push_back(id);
    }
  }

  MPICHECK(MPI_Bcast(ids.data(), sizeof(ncclUniqueId) * n_unique_ids, MPI_BYTE, 0,
                     MPI_COMM_WORLD));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&world_comm, world_size, ids[0], my_rank));
  printf("Init global communicator, size %d, rank %d\n", nranks, rank);

  if (nsub_groups > 1) {
    // local communication
    int my_local_rank = local_rank;  // avoid compiler optimization
    int my_local_size = local_size;
    int sub_group_idx = my_rank / local_size;
    ncclUniqueId local_group_id = ids[sub_group_idx + 1];
    NCCLCHECK(ncclCommInitRank(&local_comm, my_local_size, local_group_id,
                               my_local_rank));
    printf(
        "Init local communicator, local_size %d, local_rank %d\n, global_rank "
        "%d \n",
        local_size, local_rank, rank);

    // group communicator for ranks have same local_rank
    int my_group_size = nsub_groups;
    ncclUniqueId group_nccl_id = ids[1 + my_group_size + my_local_rank];
    NCCLCHECK(ncclCommInitRank(&group_comm, my_group_size, group_nccl_id,
                               my_rank / local_size));
    printf("Init cross node group communicator, group_size %d, group_rank %d\n",
           my_group_size, my_rank / local_size);
  }
  
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
  if (const char* env_p = std::getenv("NCCL_ALGO")) {
    std::string nccl_algo(env_p);
    if (nccl_algo == "Tree") {
      float n_nodes = nranks / local_size;
      ratio = 2 * (n_nodes - 1) / n_nodes;
      printf("ratio %f, nnodes %f \n", ratio, n_nodes);
    } else if (nccl_algo == "Ring") {
    } else {
      if (rank == 0)
        printf("ratio not changed by NCCL_ALGO env var %d\n", env_p);
    }
  }

  float bw = ratio * buffer_size / 1e9 / (avg_time * 1e-3);
  if (rank == 0)
    printf("flat ring allreduce on global buffer size %lu, bw %f GB/s, avg time %f ms \n", buffer_size, bw, avg_time);
}

// reduce-scatter in local groups 
// allreduce among groups, but with only sharded GPUs
// allgather in local groups
void allreduce_hierarchy(int warm_up=5, int repeat=10) {
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, stop, allreduce_start, allreduce_stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&allreduce_start);
  cudaEventCreate(&allreduce_stop);

  size_t chunk_nelem = nelem / local_size;
  size_t chunk_size = buffer_size / local_size;

  float avg_time = 0;
  float avg_allreduce_time = 0;
  for (int i = 0; i < warm_up + repeat; ++i) {
    cudaEventRecord(start, stream);
    void* send_buff = gpu_buffer;
    void* recv_buff = (char*)gpu_buffer + chunk_size * local_rank;
    // reduce scatter onto all GPUs
    NCCLCHECK(ncclReduceScatter(send_buff, recv_buff, chunk_nelem, ncclFloat32,
                                ncclSum, local_comm, stream));
    CUDACHECK(cudaEventRecord(allreduce_start, stream));
    // allreduce each chunk
    NCCLCHECK(ncclAllReduce(recv_buff, recv_buff, chunk_nelem, ncclFloat32,
                            ncclSum, group_comm, stream));
    CUDACHECK(cudaEventRecord(allreduce_stop, stream));

    // allgather allreduced results from all local GPUs
    NCCLCHECK(ncclAllGather(recv_buff, gpu_buffer, chunk_nelem, ncclFloat32,
                            local_comm, stream));

    cudaEventRecord(stop, stream);

    cudaStreamSynchronize(stream);
    float _cost = 0;
    float _allreduce_cost = 0;
    cudaEventElapsedTime(&_cost, start, stop);
    CUDACHECK(cudaEventElapsedTime(&_allreduce_cost, allreduce_start, allreduce_stop));

    if (i >= warm_up) {
      avg_time += (_cost / float(repeat));
      avg_allreduce_time += (_allreduce_cost / float(repeat));
    }
  }

  float ratio = 2 * (float(nsub_groups) - 1) / float(nsub_groups);
  float bw = ratio * buffer_size / 1e9 / (avg_allreduce_time * 1e-3);

  if (rank == 0)
    printf(
        "hierarchy: inter-node allreduce on buffer size %lu, bw %f GB/s, avg "
        "total time %f ms, avg inter-allreduce time %f ms \n",
        buffer_size, bw, avg_time, avg_allreduce_time);
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
  printf("rank %d, nranks %d, local_size %d, device_idx %d, local_rank %d, nsub %d\n", rank, nranks, local_size, device_idx, local_rank, nsub_groups);

  nelem = std::stoi(argv[2]);
  buffer_size =  nelem * sizeof(float);
  size_t chunk_nelem = nelem / local_size;
  assert(chunk_nelem * local_size == nelem); // can be equally partitioned

  init_nccl_communicators();
  init_buffers();

  allreduce_global();

  if (nsub_groups > 1)
    allreduce_hierarchy();

  ncclCommDestroy(world_comm);
  if (nsub_groups > 1) {
    ncclCommDestroy(local_comm);
    ncclCommDestroy(group_comm);
  }
  //finalizing MPI
  MPICHECK(MPI_Finalize());
}