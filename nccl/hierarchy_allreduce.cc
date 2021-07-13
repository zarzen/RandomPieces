#include "common.h"
#include "cuda_runtime.h"
#include <string>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <thread>
#include "thd_queue.h"
#include <chrono>

int rank;
int nranks;
int local_rank;
int local_size;
int device_idx;
int nsub_groups;

int size_of_dtype(ncclDataType_t dtype) {
  switch (dtype)
  {
  case ncclFloat32:
    return 4;
  case ncclFloat16:
    return 2;
  default:
    std::cerr << "unsupported data type\n";
    return -1;
    break;
  }
}

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

struct coll_task {
  void* buff;
  ncclDataType_t dtype;
  size_t count;

  void* send_buff;
  void* recv_buff;
  size_t coll_count; 
  cudaEvent_t sync_e; // after launch nccl op for each stage this sync_e will be recorded

  coll_task* parent;
  int n_sub;
  int stage; // 0: reduce-scatter; 1: all-reduce; 2: all-gather; -1 exit

  int n_complete;
  std::mutex mtx;
  std::condition_variable cv;

  coll_task() {}

  coll_task(void* buff, ncclDataType_t dtype, size_t nelem, int n_sub)
      : buff(buff),
        dtype(dtype),
        count(nelem),
        n_sub(n_sub),
        stage(0),
        n_complete(0) {
    CUDACHECK(cudaEventCreate(&sync_e));
  }
  // wait for all sub tasks
  void wait() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [this] { return this->n_complete == this->n_sub; });
  }

  void complete_one_sub() {
    {
      std::lock_guard<std::mutex> lk(mtx);
      ++n_complete;
    }
    cv.notify_all();
  }

  // different stage has different send/recv buffer
  void compute_send_recv(int local_rank, int local_size) {
    int coll_buff_size = this->coll_count * size_of_dtype(this->dtype);
    switch (this->stage) {
      case 0:  // intra reduce scatter
        this->send_buff = this->buff;
        this->coll_count = this->count / local_size;
        this->recv_buff = (char*)this->buff + coll_buff_size * local_rank;
        break;
      case 1:                                         // inplace all-reduce
        this->coll_count = this->count / local_size;  // assume dividable
        this->send_buff = (char*)this->buff + coll_buff_size * local_rank;
        this->recv_buff = this->send_buff;
        break;
      case 2:
        this->coll_count = this->count / local_size;
        this->send_buff = (char*)this->buff + coll_buff_size * local_rank;
        this->recv_buff = this->buff;
        break;
      default:
        break;
    }
  }
};

void launch_coll(coll_task* t, ncclComm_t comm, cudaStream_t stream) {
  CUDACHECK(cudaSetDevice(device_idx)); 
  // int dev;
  // cudaGetDevice(&dev);
  // printf("cuda current dev %d\n", dev);
  t->compute_send_recv(local_rank, local_size);
  if (rank == 0)
    printf("send_buf %p, recv_buf %p, coll_count %lu, dtype %d, cudaStream %d \n", t->send_buff,
           t->recv_buff, t->coll_count, t->dtype, stream);
  switch(t->stage) {
    case 0:
      NCCLCHECK(ncclReduceScatter(t->send_buff, t->recv_buff, t->coll_count, t->dtype,
                                  ncclSum, comm, stream));
      break;
    case 1:
      NCCLCHECK(ncclAllReduce(t->send_buff, t->recv_buff, t->coll_count, t->dtype,
                              ncclSum, comm, stream));
      break;
    case 2:
      NCCLCHECK(ncclAllGather(t->send_buff, t->recv_buff, t->coll_count, t->dtype,
                              comm, stream));
      break;

  }
  CUDACHECK(cudaEventRecord(t->sync_e, stream));
}

/*
  intra node loop takes care of reduce-scatter and all-gather
*/
void intra_node_loop(ThdSafeQueue<coll_task*>& rs_tasks,
                     ThdSafeQueue<coll_task*>& ar_tasks,
                     ThdSafeQueue<coll_task*>& ag_tasks,
                     ncclComm_t intra_comm,
                     cudaStream_t intra_stream) {
  coll_task* t;
  std::vector<coll_task*> ongoing_rs;
  std::vector<coll_task*> ongoing_ag;
  while (true) {
    rs_tasks.pop(&t);
    // exit check
    if (t->stage == -1)
      return;

    // intra node reduce_scatter launch
    for (int i = 0; i < t->parent->n_sub; ++i) {
      if(i > 0) // i == 0 -> already poped
        rs_tasks.pop(&t);

      launch_coll(t, intra_comm, intra_stream);
      ongoing_rs.push_back(t);
    }

    // sync intra node reduce scatter than launch inter node op
    for (int i = 0; i < t->parent->n_sub; ++i){
      CUDACHECK(cudaEventSynchronize(ongoing_rs[i]->sync_e));
      ongoing_rs[i]->stage++;
      ar_tasks.push(ongoing_rs[i]);
    }

    // start fetching tasks for intra all-gather
    for (int i = 0; i < t->parent->n_sub; ++i) {
      ag_tasks.pop(&t);
      launch_coll(t, intra_comm, intra_stream);
      ongoing_ag.push_back(t);
    }

    // sync intra all-gather
    for (int i = 0; i < t->parent->n_sub; ++i) {
      CUDACHECK(cudaEventSynchronize(ongoing_ag[i]->sync_e));
      ongoing_ag[i]->parent->complete_one_sub();
    }
  }
}

void inter_node_loop(ThdSafeQueue<coll_task*>& ar_tasks,
                     ThdSafeQueue<coll_task*>& ag_tasks,
                     ncclComm_t inter_comm,
                     cudaStream_t inter_stream) {
  coll_task* t;
  std::vector<coll_task*> ongoing_ar;
  while (true) {
    ar_tasks.pop(&t);
    if (t->stage == -1) 
      return;
    
    // launch task
    for (int i = 0; i < t->parent->n_sub; ++i) {
      if (i > 0)
        ar_tasks.pop(&t);
      launch_coll(t, inter_comm, inter_stream);
      ongoing_ar.push_back(t);
    }

    // sync and launch allgather
    for (int i = 0; i < t->parent->n_sub; ++i) {
      CUDACHECK(cudaEventSynchronize(ongoing_ar[i]->sync_e));
      ongoing_ar[i]->stage++; // move to next stage
      ag_tasks.push(ongoing_ar[i]);
    }
  }
}

size_t chunk_nelem = 1000000; // 1M elements floats-> 4MB
/*
  using thread with queue to async launch tasks
*/
void pipelined_hierarchy(int warm_up=5, int repeat=10) {
  // prepare task queues
  ThdSafeQueue<coll_task*> rs_tasks;
  ThdSafeQueue<coll_task*> ar_tasks;
  ThdSafeQueue<coll_task*> ag_tasks;

  cudaStream_t intra_stream;
  cudaStream_t inter_stream;
  CUDACHECK(cudaStreamCreate(&intra_stream));
  CUDACHECK(cudaStreamCreate(&inter_stream));
  if (rank == 0) 
    printf("intra stream %d, inter stream %d\n", intra_stream, inter_stream);

  std::thread intra_thd =
      std::thread(intra_node_loop, std::ref(rs_tasks), std::ref(ar_tasks),
                  std::ref(ag_tasks), local_comm, intra_stream);
  std::thread inter_thd =
      std::thread(inter_node_loop, std::ref(ar_tasks), std::ref(ag_tasks),
                  group_comm, inter_stream);

  float avg_time = 0;
  for (int i = 0; i < warm_up + repeat; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    // create parent task and sub tasks
    coll_task task(gpu_buffer, ncclFloat32, nelem, nelem / chunk_nelem);
    if (rank == 0)
      printf("task buff %p, dtype %d \n", task.buff, ncclFloat32);

    std::vector<coll_task*> sub_tasks;
    size_t elem_offset = 0;
    for (int k = 0; k < task.n_sub; ++k) {
      void* chunk_buff = (char*)task.buff + elem_offset * size_of_dtype(task.dtype);
      size_t count = (k == task.n_sub - 1) ? nelem - elem_offset : chunk_nelem;
      coll_task* sub_t = new coll_task(chunk_buff, task.dtype, count, 0);
      sub_t->parent = &task;
      sub_t->buff = chunk_buff;
      elem_offset += chunk_nelem;
      if (rank == 0)
        printf("sub task buff %p, count %lu, dtype %d \n", sub_t->buff,
               sub_t->count, sub_t->dtype);

      rs_tasks.push(sub_t);
      sub_tasks.push_back(sub_t); // just for record
    }

    task.wait();
    for (auto t : sub_tasks) {
      delete t;
    }

    auto end = std::chrono::high_resolution_clock::now();
    // record time
    
    std::chrono::duration<double> elapsed_seconds = end-start;
    if (i >= warm_up) {
      avg_time += elapsed_seconds.count() / repeat;
    }
  }

  float ratio = 2 * (float(nsub_groups) - 1) / float(nsub_groups);
  float bw = ratio * buffer_size / 1e9 / (avg_time * 1e-3);
  if (rank == 0)
    printf(
        "pipelined hierarchy: inter-node allreduce on buffer size %lu, bw %f "
        "GB/s, avg "
        "total time %f ms \n",
        buffer_size, bw, avg_time);

  // at the end push a task with stage -1
  coll_task exit_task;
  exit_task.stage = -1;
  rs_tasks.push(&exit_task);
  ar_tasks.push(&exit_task);
  ag_tasks.push(&exit_task);

  intra_thd.join();
  inter_thd.join();
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

  if (nsub_groups > 1)
    pipelined_hierarchy();

  ncclCommDestroy(world_comm);
  if (nsub_groups > 1) {
    ncclCommDestroy(local_comm);
    ncclCommDestroy(group_comm);
  }
  //finalizing MPI
  MPICHECK(MPI_Finalize());
}