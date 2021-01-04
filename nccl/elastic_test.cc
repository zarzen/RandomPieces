#include "nccl.h"
#include "sdcc_tcp.h"
#include "mpi.h"
#include <memory>
#include <atomic>
#include <cassert>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>

using std::shared_ptr;
using std::atomic;
using std::unordered_map;

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      throw "mpi check failed";                                        \
    }                                                                  \
  } while (0)

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      throw "cuda check failed";                                    \
    }                                                               \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t rank = cmd;                                        \
    if (rank != ncclSuccess) {                                      \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(rank));                             \
      throw "nccl error";                                           \
    }                                                               \
  } while (0)

void ipStringToInts(std::string& ip, int* ret) {
  std::istringstream is(ip);
  std::string i;
  int idx = 0;
  while (std::getline(is, i, '.')){
    ret[idx] = std::stoi(i);
    ++idx;
  }
}

std::string ipStr(int* ips){
  std::stringstream ss;
  ss << ips[0] << "." << ips[1] << "." << ips[2] << "." << ips[3];
  return ss.str();
}

typedef enum {
  Ping = 0,
  Shutdown = 1,
  Reset = 2,
  PushNcclId = 3,
  SendWorkerInfo = 4,
  Pause = 5,
} ctrl_type_t;

struct WorkerInfo {
  int world_size = -1;
  int rank = -1;
  int local_rank = -1;
  int worker_ip[4] = {0, 0, 0, 0};
  int listen_port = 0;
  bool is_you = false;
};

struct WorkerStates {
  ncclComm_t nccl_comm;
  cudaStream_t cuda_stream;
  bool init_once = false;
  bool control_link_ready = false;
  bool self_info_ready = false;
  bool memory_ready = false;
};

struct CtrlMsg {
  // PING, DOWN(shutdown), RSET (reset), PHID(push id), 
  ctrl_type_t type;
  union {
    WorkerInfo worker_info;
    ncclUniqueId nccl_id;
  };
  CtrlMsg() {}
};

// ************** inital setup
// has some initial setup for system to launch
// put the value in main function
std::vector<WorkerInfo> initial_setup;
// ************** end the inital setup

int getIpIndex(std::string& ip, std::vector<WorkerInfo>& workers);

// listen for workers to connect
// reply with the rank, local rank, all other node informations
// the first stage the control loop wait for all nodes to join
// after some time controller send the "reset" signal to all workers
// it pick one node to exit
// thus, world_size, pre_worker, and next_worker of a worker might change
// and, the NCCL communicator requires update
void workerControllerLoop(int listen_port) {
  int BASE_PORT = 10000;
  shared_ptr<sdcc::TCPServer> server =
      std::make_shared<sdcc::TCPServer>("0.0.0.0", listen_port, 100, true);
  std::cout << "start controller server " << listen_port << "\n";

  std::vector<shared_ptr<sdcc::TCPAgent>> workers;
  workers.reserve(initial_setup.size());
  workers.resize(initial_setup.size());
  // first stage wait for all workers
  for (int i = 0; i < initial_setup.size(); ++i) {
    // accept the client, and get their ip, then match with initial setup
    shared_ptr<sdcc::TCPAgent> w_conn = server->acceptCli(true);

    std::string cli_ip = w_conn->getIP();
    // match ip and give port 
    int rank = getIpIndex(cli_ip, initial_setup);
    if (rank == -1) {
      std::cerr << __FILE__ << ":" << __LINE__ << ": getIpIndex error\n";
    }
    std::cout << "get connection from ip " << cli_ip << " rank " << rank
              << " fd " << w_conn->getFd() << '\n';

    int port = BASE_PORT + rank;
    initial_setup[rank].listen_port = port;
    workers[rank] = w_conn;
  }
  std::cout << "worker controller pass first stage, workers size "
            << workers.size() << "\n";

  // second stage send the port back to worker, for them to start the listen
  for (int i = 0; i < initial_setup.size(); ++i) {
    std::shared_ptr<sdcc::TCPAgent>& w_conn = workers[i];

    WorkerInfo w_info = initial_setup[i];
    w_info.is_you = true;
    CtrlMsg msg;
    msg.worker_info = w_info;
    msg.type = ctrl_type_t::Reset;

    w_conn->send((char*)&msg, sizeof(msg));
    std::cout << "sent worker rank " << msg.worker_info.rank << "\n";
  }
  std::cout << "worker controller passed second stage\n";

  // third stage send other workers info to each worker
  for (int i = 0; i < initial_setup.size(); ++i) {
    shared_ptr<sdcc::TCPAgent>& w_conn = workers[i];

    for (int j = 0; j < initial_setup[i].world_size; ++j) {
      if (j != i) {
        WorkerInfo w_info = initial_setup[j];
        w_info.is_you = false;
        CtrlMsg msg;
        msg.type = ctrl_type_t::SendWorkerInfo;
        msg.worker_info = w_info;

        w_conn->send((char*)&msg, sizeof(msg));
      }
    }
  }
  std::cout << "send other workers info to each worker\n";

  // third + additional wait for NcclUniqueId from rank 0
  // then broadcast to others
  {
    shared_ptr<sdcc::TCPAgent>& rank0_conn = workers[0];
    CtrlMsg msg;
    std::cout << "wait receive ncclUniqueId from rank 0\n";
    rank0_conn->recv((char*)&msg, sizeof(msg));
    assert(msg.type == ctrl_type_t::PushNcclId);
    for (int i = 1; i < initial_setup.size(); ++i) {
      shared_ptr<sdcc::TCPAgent> w_conn = workers[i];
      w_conn->send((char*)&msg, sizeof(msg));
    }
  }
  std::cout << "worker controller passed third stage\n";

  std::atomic<bool> _ping_stop(false);
  std::thread _ping_thread([&workers, &_ping_stop](){
    std::cout << "workers size " << workers.size() << "\n";
    while(!_ping_stop) {
      CtrlMsg msg;
      msg.type = ctrl_type_t::Ping;
      for (int i = 0; i < workers.size(); ++i) {
        workers[i]->send(&msg, sizeof(msg));
      }
      for (int i = 0; i < workers.size(); ++i) {
        workers[i]->recv(&msg, sizeof(msg));
        assert(msg.type == ctrl_type_t::Ping);
      }
      std::cout << "1 send&recv ping\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  // fourth stage wait for certain time then drop one worker
  std::this_thread::sleep_for(std::chrono::seconds(10));
  _ping_stop = true;
  _ping_thread.join();
  std::cout << "worker controller after first sleep\n";

  // fifth stage pause all workers
  for (int i = 0; i < initial_setup.size(); ++i) {
    shared_ptr<sdcc::TCPAgent>& w_conn = workers[i];
    
    CtrlMsg msg;
    msg.type = ctrl_type_t::Pause;
    w_conn->send((char*)&msg, sizeof(msg));
  }
  std::cout << "worker controller passed send Pause\n";

  // sixth stage drop the last worker, reset world_info of the rest workers
  int drop_idx = initial_setup.size() - 1;
  // shut down the drop node
  shared_ptr<sdcc::TCPAgent> drop_w_conn = workers[drop_idx];
  CtrlMsg msg;
  msg.type = ctrl_type_t::Shutdown;
  drop_w_conn->send(&msg, sizeof(msg));
  std::cout << "send shutdown to ip " << drop_w_conn->getIP() << '\n';
  // update workers info
  std::vector<WorkerInfo> updated_infos;
  std::vector<shared_ptr<sdcc::TCPAgent>> updated_conns;
  for (int i = 0; i < initial_setup.size(); ++i) {
    if (i != drop_idx) {
      WorkerInfo info = initial_setup[i];
      updated_infos.push_back(info);
      updated_conns.push_back(workers[i]);
    }
  }
  for (int i = 0; i < updated_infos.size(); ++i) {
    updated_infos[i].world_size--;
    updated_infos[i].rank = i;
    updated_infos[i].listen_port = BASE_PORT + i;
    // ip and local_rank do not change
    std::cout << "update info " << i << " \n ";
  }
  std::cout << "computed updated infos size " << updated_infos.size() << "\n";
  // send reset and info of each node
  for (int i = 0; i < updated_infos.size(); ++i) {
    shared_ptr<sdcc::TCPAgent>& w_conn = updated_conns[i];
    if (w_conn == nullptr) {
      std::cerr << "w_conn " << i << " is nullptr\n";
    }
    CtrlMsg msg;
    msg.type = ctrl_type_t::Reset;
    WorkerInfo w_info = updated_infos[i];
    w_info.is_you = true;  // update self info
    msg.worker_info = w_info;

    w_conn->send(&msg, sizeof(msg));
  }
  std::cout << "send out updated infos\n";
  // send other node info
  for(int i = 0; i < updated_infos.size(); ++i) {
    shared_ptr<sdcc::TCPAgent>& w_conn = updated_conns[i];
    for (int j = 0; j < updated_infos.size(); ++j) {
      if (j != i) { // is other node
        WorkerInfo w_info = updated_infos[j];
        w_info.is_you = false;
        CtrlMsg msg;
        msg.type = ctrl_type_t::SendWorkerInfo;
        msg.worker_info = w_info;

        w_conn->send(&msg, sizeof(msg));
        std::cout << "send to " << i << ", " << j << "\n";
      }
    }
  }

  // wait for rank0 to broadcase new ncclUniqueId
  {
    shared_ptr<sdcc::TCPAgent>& rank0_conn = updated_conns[0];
    CtrlMsg msg;
    rank0_conn->recv(&msg, sizeof(msg));
    for (int i = 1; i < updated_infos.size(); ++i) {
      shared_ptr<sdcc::TCPAgent>& w_conn = updated_conns[i];
      w_conn->send(&msg, sizeof(msg));
    }
  }
  std::cout << "worker controller broadcast ncclUniqueId\n";

  atomic<bool> _sec_ping_stop(false);
  std::thread _sec_ping([&updated_conns, &_sec_ping_stop](){
    while(!_sec_ping_stop) {
      CtrlMsg msg;
      msg.type = Ping;
      for (int i = 0; i<updated_conns.size(); ++i) {
        updated_conns[i]->send(&msg, sizeof(msg));
      }
      for (int i=0; i < updated_conns.size(); ++i) {
        updated_conns[i]->recv(&msg, sizeof(msg));
        assert(msg.type == Ping);
      }
      std::cout << "2 send&recv ping\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  // wait for another seconds to shutdown all nodes
  std::this_thread::sleep_for(std::chrono::seconds(5));
  _sec_ping_stop = true;
  _sec_ping.join();
  std::cout <<"worker controller after second sleep\n";

  for (int i = 0; i < updated_conns.size(); ++i) {
    CtrlMsg msg;
    msg.type = ctrl_type_t::Shutdown;
    updated_conns[i]->send(&msg, sizeof(msg));
  }
  std::cout << "worker controller send out Shutdown\n";
}

void resetNcclComm(shared_ptr<sdcc::TCPClient>& to_controller,
                   int world_size,
                   int rank,
                   int local_rank,
                   ncclComm_t* comm,
                   bool init_before) {
  ncclResult_t result;
  if (init_before) {
    ncclCommDestroy(*comm);
  }
  if (rank == 0) {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    CtrlMsg msg;
    msg.type = PushNcclId;
    msg.nccl_id = id;
    to_controller->send((char*)&msg, sizeof(msg));
    std::cout << "rank 0 sent out the ncclUniqueId " << "\n";
    cudaSetDevice(local_rank);
    result = ncclCommInitRank(comm, world_size, id, rank);
  } else {
    CtrlMsg msg;
    to_controller->recv((char*)&msg, sizeof(msg));
    assert(msg.type == PushNcclId);
    cudaSetDevice(local_rank);
    std::cout << "received ncclUniqueId " << " local rank "
              << local_rank << "\n";
    result = ncclCommInitRank(comm, world_size, msg.nccl_id, rank);
  }
  if (result != ncclSuccess) {
    printf("resetNcclComm error");
  }
}

void acceptPreWorker(shared_ptr<sdcc::TCPServer>& local_server, shared_ptr<sdcc::TCPAgent>& tcp_from_pre) {
  tcp_from_pre = local_server->acceptCli(true);
  std::cout << "accepted pre_worker connection \n";
}

void resetConnections(CtrlMsg& msg,
                      shared_ptr<sdcc::TCPClient>& to_controller,
                      WorkerInfo& self_info,
                      unordered_map<int, WorkerInfo>& other_workers,
                      shared_ptr<sdcc::TCPServer>& local_server,
                      shared_ptr<sdcc::TCPAgent>& tcp_to_next,
                      shared_ptr<sdcc::TCPAgent>& tcp_from_pre,
                      WorkerStates& state) {
  assert(msg.worker_info.is_you == true);
  self_info = msg.worker_info;
  state.self_info_ready = true;

  // if (local_server != nullptr)
  //   local_server.reset();  // close current server
  if (local_server != nullptr && local_server->getPort() == self_info.listen_port) {
  } else {
    local_server = std::make_shared<sdcc::TCPServer>(
      "0.0.0.0", self_info.listen_port, 10, true);
  }

  std::thread _local_thd(acceptPreWorker, std::ref(local_server),
                         std::ref(tcp_from_pre));
  std::cout << "started local server at port " << self_info.listen_port << "\n";
  other_workers.clear();
  // receive 
  for (int i = 0; i < self_info.world_size - 1; ++i) {
    CtrlMsg other_w_info;
    to_controller->recv(&other_w_info, sizeof(other_w_info));
    assert(other_w_info.type == ctrl_type_t::SendWorkerInfo);
    other_workers[other_w_info.worker_info.rank] = other_w_info.worker_info;
    std::cout << "received other worker rank " << other_w_info.worker_info.rank
              << "\n";
  }

  int next_rank = (self_info.rank + 1) % self_info.world_size;
  // build tcp connection to next worker
  WorkerInfo& next_worker = other_workers[next_rank];
  tcp_to_next = std::make_shared<sdcc::TCPClient>(
      ipStr(next_worker.worker_ip), next_worker.listen_port, true);
  std::cout << "tcp conn to next worker established\n";
  _local_thd.join();
  state.control_link_ready = true;
  std::cout << "tcp control with pre and next node estabilished\n";
  double _start_t = std::chrono::high_resolution_clock::now()
                                  .time_since_epoch()
                                  .count() /
                              1e6;
  resetNcclComm(to_controller, self_info.world_size, self_info.rank,
                self_info.local_rank, &(state.nccl_comm), state.init_once);
  double _end_t = std::chrono::high_resolution_clock::now()
                                  .time_since_epoch()
                                  .count() /
                              1e6;
  std::cout << "setup nccl communicator cost (ms) " << _end_t - _start_t << "\n";
  state.init_once = true;
}

// listen for command info from controller
// ping: check the status, reply with ping
// reset: receive following information about the world, and build the ncclComm_t
// shutdown: exit
bool checkControlMessage(shared_ptr<sdcc::TCPClient>& to_controller,
                         atomic<bool>& exit,
                         WorkerInfo& self_info,
                         unordered_map<int, WorkerInfo>& other_workers,
                         shared_ptr<sdcc::TCPServer>& local_server, 
                         shared_ptr<sdcc::TCPAgent>& tcp_to_next,
                         shared_ptr<sdcc::TCPAgent>& tcp_from_pre,
                         WorkerStates& states) {
  CtrlMsg msg;
  int received_bytes = to_controller->irecv(&msg, sizeof(msg));
  // to_controller->recv(&msg, sizeof(msg));
  
  if (received_bytes > 0) {
    // std::cout << "received bytes " << received_bytes << "\n";
    if (received_bytes < sizeof(msg))
      to_controller->recv((&msg) + received_bytes, sizeof(msg) - received_bytes);
    // std::cout << "received msg type " << msg.type << "\n";
    switch(msg.type) {
      case ctrl_type_t::Shutdown:
        exit = true;
        std::cout << "received shutdown signal\n";
        return false;
      case ctrl_type_t::Ping:
        // std::cout << "received Ping\n";
        to_controller->send(&msg, sizeof(msg));
        return true;
      case ctrl_type_t::Pause:
        // followed by reset
        std::cout << "received Pause\n";
        // reset local_server
        tcp_to_next.reset();
        tcp_from_pre.reset();
        to_controller->recv(&msg, sizeof(msg));
        if (msg.type == ctrl_type_t::Shutdown) {
          exit = true;
          std::cout << "rank " << self_info.rank << " exiting \n";
          return false;
        }
      case ctrl_type_t::Reset: {
        states.self_info_ready = false;
        states.control_link_ready = false;
        std::cout << "received reset msg self_info rank " << msg.worker_info.rank << "\n";
        double _start_reset = std::chrono::high_resolution_clock::now()
                                  .time_since_epoch()
                                  .count() /
                              1e6;
        resetConnections(msg, to_controller, self_info, other_workers,
                         local_server, tcp_to_next, tcp_from_pre, states);
        double _end_reset = std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count() /
                            1e6;
        std::cout << "reset connection cost (ms)" << _end_reset - _start_reset
                  << "\n";
      } return false; break;
      default:
        std::cerr << "unhandled clause msg type" << msg.type << std::endl;
        break;
    }
  } else {
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return false;
  }
}

// mock the send receiving operations in nccl
bool ncclSendRecvOnce(WorkerStates& states,
                      WorkerInfo& self_info,
                      void* send_buff,
                      size_t send_count,
                      void* recv_buff,
                      size_t recv_count,
                      shared_ptr<sdcc::TCPAgent>& to_next,
                      shared_ptr<sdcc::TCPAgent>& from_pre,
                      int wait_ms_after) {
  int next_rank = (self_info.rank + 1) % self_info.world_size;
  int pre_rank =
      (self_info.rank - 1 + self_info.world_size) % self_info.world_size;
  bool ret = true;
  if (!states.control_link_ready) {
    // std::cout << "control link not ready\n";
    return false;
  }
  try {
    // mock send receive through tcp control link for meta data exchange;
    std::thread send_thd([&to_next](){
      uint64_t meta[3] = {0,0,0};
      to_next->send(meta, sizeof(meta));
    });

    std::thread recv_thd([&from_pre](){
      uint64_t meta[3] = {0,0,0};
      from_pre->recv(meta, sizeof(meta));
    });
    send_thd.join();
    recv_thd.join();
    // std::cout << "send receive meta data done\n";

    CUDACHECK(cudaSetDevice(self_info.local_rank));
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(send_buff, send_count, ncclFloat32, next_rank, states.nccl_comm, states.cuda_stream));
    NCCLCHECK(ncclRecv(recv_buff, recv_count, ncclFloat32, pre_rank, states.nccl_comm, states.cuda_stream));
    NCCLCHECK(ncclGroupEnd());

    CUDACHECK(cudaStreamSynchronize(states.cuda_stream));
    // std::cout << "send recv complete once\n";
  } catch (const std::exception& e) {
    std::cerr << "nccl runing into error " << e.what() << "\n";
    ret = false;
  }

  if (wait_ms_after > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms_after));
  }

  return ret;
}

void basic_tests();
void workerInitSetup();

void initCudaMemory(WorkerInfo& self_info,
                    WorkerStates& state,
                    void*& send_buff,
                    size_t send_count,
                    void*& recv_buff,
                    size_t recv_count) {
  CUDACHECK(cudaSetDevice(self_info.local_rank));
  CUDACHECK(cudaMalloc(&send_buff, send_count * sizeof(float)));
  CUDACHECK(cudaMalloc(&recv_buff, recv_count * sizeof(float)));
  CUDACHECK(cudaMemset(recv_buff, 0, recv_count * sizeof(float)));
  CUDACHECK(cudaMemset(send_buff, 1, send_count * sizeof(float)));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  state.cuda_stream = stream;

  CUDACHECK(cudaStreamSynchronize(state.cuda_stream));
  std::cout << "after init send buff ptr " << send_buff << "\n";
}

int main(int argc, char* argv[]) {
  workerInitSetup();
  // basic_tests();
  // setup memory and get initial mpi world and rank
  int rank, nranks;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  if (argc < 2) {
    std::cerr << "require input of worker controller (at rank0) ip\n";
    return -1;
  }
  std::string worker_controller_ip(argv[1]);

  int controller_port = 5555;
  shared_ptr<std::thread> controller_thread;
  if (rank == 0) {
    controller_thread = std::make_shared<std::thread>(workerControllerLoop, controller_port);
  }

  void* send_buff;
  size_t send_count = 1 * 1024 * 1024;
  void* recv_buff;
  size_t recv_count = 1 * 1024 * 1024;

  float* host_buff = new float[send_count];

  atomic<bool> exit(false);
  // connect to controller
  shared_ptr<sdcc::TCPClient> to_controller;
  to_controller = std::make_shared<sdcc::TCPClient>(worker_controller_ip,
                                                    controller_port, true);

  WorkerStates states;
  WorkerInfo self_info;

  unordered_map<int, WorkerInfo> other_workers;
  shared_ptr<sdcc::TCPServer> local_server;
  shared_ptr<sdcc::TCPAgent> to_next;
  shared_ptr<sdcc::TCPAgent> from_pre;
  int c = 0;
  while (!exit) {
    bool check_res = checkControlMessage(to_controller, exit, self_info, other_workers,
                        local_server, to_next, from_pre, states);
    if (check_res && states.init_once && !states.memory_ready) {
      std::cout << "initing memory\n";
      initCudaMemory(self_info, states, send_buff, send_count, recv_buff,
                     recv_count);

      states.memory_ready = true;

      std::cout << "send buff " << (void*) send_buff << "\n";
    }
    if (check_res && states.memory_ready && !exit && states.self_info_ready){
      
      CUDACHECK(cudaMemcpy(send_buff, host_buff, send_count * sizeof(float),
                           ::cudaMemcpyDefault));
      CUDACHECK(cudaMemcpy(recv_buff, host_buff, send_count * sizeof(float),
                           ::cudaMemcpyDefault));
      bool res = ncclSendRecvOnce(states, self_info, send_buff, send_count,
                                  recv_buff, recv_count, to_next, from_pre, 0);
      if (!res) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      } else {
        // std::cout << "completed send recv " << c++ << "\n";
      }
    }
  }

  std::cout << "exit\n";

  if (rank == 0 && controller_thread->joinable()) {
    controller_thread->join();
    std::cout << "joined controller\n";
  }
}

void basic_tests() {
  // printf("size of ctrl msg %zu\n", sizeof(CtrlMsg));
  for (auto a : initial_setup) {
    printf("%d.%d.%d.%d:%d\n", a.worker_ip[0], a.worker_ip[1], a.worker_ip[2],
           a.worker_ip[3], a.listen_port);
  }

  std::string ip = "172.31.88.106";
  int test[4];
  ipStringToInts(ip, test);
  std::cout << ipStr(test) << std::endl;
}

void workerInitSetup() {
  // **************** initialize worker info setup
  WorkerInfo worker0; 
  worker0.world_size = 3;
  worker0.rank = 0;
  worker0.local_rank = 0;
  int ip0[4] = {172,31,88,106};
  memcpy(worker0.worker_ip, ip0, 4 * sizeof(int));
  worker0.listen_port = 0; // init port all zeros, the control loop will set it

  WorkerInfo worker1;
  worker1.world_size = 3;
  worker1.rank = 1;
  worker1.local_rank = 0;
  int ip1[4] = {172,31,76,23};
  memcpy(worker1.worker_ip, ip1, 4 * sizeof(int));
  worker1.listen_port = 0;

  WorkerInfo worker2;
  worker2.world_size = 3;
  worker2.rank = 2;
  worker2.local_rank = 0;
  int ip2[4] = {172,31,79,244};
  memcpy(worker2.worker_ip, ip2, 4 * sizeof(int));
  worker2.listen_port = 0;

  initial_setup.push_back(worker0);
  initial_setup.push_back(worker1);
  initial_setup.push_back(worker2);

  // **************** end init setup
}

int getIpIndex(std::string& ip, std::vector<WorkerInfo>& workers){
  int ip_ints[4];
  ipStringToInts(ip, ip_ints);
  int idx = 0;
  for (auto& w:workers) {
    if (memcmp(w.worker_ip, ip_ints, sizeof(ip_ints)) == 0) {
      return idx;
    }
    ++idx;
  }
  return -1;
}