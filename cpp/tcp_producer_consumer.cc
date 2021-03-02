#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sstream>
#include "logger.h"
#include <vector>
#include <queue>
#include <mutex>
#include <chrono>
#include <thread>
#include <unordered_map>


double timeMs() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count() /
         1e6;
};

template<typename T>
static inline void fillVals(T* buff, size_t count) {
  srand(123);
  for (int i = 0; i < count; ++i) {
    T e = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    buff[i] = e;
  }
}

double floatSummary(float* buff, int nelem) {
  double sum = 0;
  for (int i = 0; i < nelem; ++i) {
    sum += *(buff + i);
  }
  return sum;
}

void* send_buff;

bool createListenSocket(int* fd, int port) {
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  LOG_IF_ERROR(socket_fd == -1, "open socket error");

  int opt = 1;
  int ret;
  ret = setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt));
  LOG_IF_ERROR(ret == -1, "setsockopt failed");

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  ret = ::bind(socket_fd, (struct sockaddr*)&addr, sizeof(addr));
  LOG_IF_ERROR(ret < 0, "socket bind to port %d failed", port);

  LOG_IF_ERROR(listen(socket_fd, 16384) < 0, "socket listen failed");
  *fd = socket_fd;
  return true;
}

int socketAccept(int& server_fd, bool tcp_no_delay) {
  int cli_fd;
  struct sockaddr_in addr;
  auto addr_len = sizeof(addr);
  cli_fd = accept(server_fd, (struct sockaddr*)&addr, (socklen_t*)&addr_len);
  LOG_IF_ERROR(cli_fd < 0, "accept client failed, server fd %d", server_fd);

  if (tcp_no_delay) {
    int opt = 1;
    auto ret = setsockopt(cli_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    LOG_IF_ERROR(ret < 0, "enabling TCP_NODELAY failed");
  }
  return cli_fd;
}

std::string ipIntsToStr(int* p, int n) {
  std::stringstream ss;
  for (int i = 0; i < n; ++i) {
    ss << p[i];
    if (i != n-1) ss << ".";
  }
  return ss.str();
}

// TODO: 
int createSocketClient(int* ip, int port, bool no_delay) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  LOG_IF_ERROR(fd == 0, "create socket fd failed");

  if (no_delay) {
    int opt = 1;
    auto ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    LOG_IF_ERROR(ret < 0, "enabling TCP_NODELAY failed");
  }

  struct sockaddr_in serv_addr;
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);
  std::string server_ip = ipIntsToStr(ip, 4);
  int ret = inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr);
  LOG_IF_ERROR(ret <= 0, "converting ip addr failed");

  bool retry = false;
  int MAX_RETRY = 10000;
  int retry_count = 0;
  do {
    int ret = ::connect(fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    if (ret != 0) retry = true;
    retry_count++;
    LOG_IF_ERROR((ret != 0 && retry_count % 1000 == 0), "connecting returned %s, retrying", strerror(errno));
  } while (retry && retry_count < MAX_RETRY);

  LOG_IF_ERROR(retry, "connect to %s:%d failed: %s", server_ip.c_str(), port, strerror(errno));
  return fd;
}

void ipStrToInts(std::string& ip, int* ret) {
  std::istringstream is(ip);
  std::string i;
  int idx = 0;
  while (std::getline(is, i, '.')){
    ret[idx] = std::stoi(i);
    ++idx;
    if (idx >= 4) return;
  }
}

#define N_DATA_SOCK 16
#define SOCK_REQ_SIZE (4*1024 * 1024) // 512kB or 1MB
#define SOCK_TASK_SIZE (128 * 1024) // 64kB
// #define N_SOCK_REQ 4 // 4 slots 
#define MAX_TASKS (2 * 1024) // for test only

struct SocketTask {
  void* ptr;
  int size;
  int stage;
  int exp_id;
};

struct FakeControlData {
  int exp_id;
  int exit;
  char pad[8];
};


void sendThread(int fd, std::queue<SocketTask*>& task_queue, std::mutex& mtx, bool& exit) {

  SocketTask* task = nullptr;
  FakeControlData ctrl_msg;

  FakeControlData ctrl2;

  int send_count = 0;

  while (!exit) {
    if (!task_queue.empty()) {
      { 
        std::lock_guard<std::mutex> lk(mtx);
        if (!task_queue.empty()) {
          task = task_queue.front();
          task_queue.pop();
        }
      }

      if (task != nullptr) {
        ctrl_msg.exp_id = task->exp_id;
        LOG_IF_ERROR(
            ::send(fd, &ctrl_msg, sizeof(ctrl_msg), MSG_WAITALL) != sizeof(ctrl_msg),
            "send ctrl msg error");

        double s = timeMs();
        int ret = ::send(fd, task->ptr, task->size, MSG_WAITALL);
        LOG_IF_ERROR(ret != task->size, "send data failed");

        // LOG_IF_ERROR(::recv(fd, &ctrl2, sizeof(ctrl2), MSG_WAITALL) != sizeof(ctrl2), "failed at recv confirmation");

        task->stage = 2;
        // double e = timeMs();
        // LOG_DEBUG("fd %d, size %d, bw %f Gbps", fd, task->size, task->size * 8 / (e - s) / 1e6);

        task = nullptr;
        send_count ++;
      }
    }
  }
  ctrl_msg.exit = 1;
  LOG_IF_ERROR(::send(fd, &ctrl_msg, sizeof(ctrl_msg), 0) != sizeof(ctrl_msg),
               "send ctrl msg error");
  LOG_DEBUG("send fd %d, send count %d", fd, send_count);
}

#define N_EXP 100

void serverMode(int port) {
  int listen_fd;
  bool ret = createListenSocket(&listen_fd, port);
  LOG_IF_ERROR(ret == false, "create listen server failed");

  int ctrl_fd = socketAccept(listen_fd, true);

  std::vector<int> data_fds;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    int fd = socketAccept(listen_fd, false);
    data_fds.push_back(fd);
  }
  LOG_DEBUG("serv build data links at port %d", port);

  // experiments send message to client in producer consumer 
  SocketTask tasks[MAX_TASKS];
  std::queue<SocketTask*> task_queue;
  std::mutex task_mtx;

  // void* buffer = malloc(SOCK_REQ_SIZE);
  void* buffer = send_buff;
  int n_tasks = SOCK_REQ_SIZE / SOCK_TASK_SIZE;

  std::vector<std::thread> background_threads;
  bool exit = false;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    background_threads.emplace_back(sendThread, data_fds[i],
                                    std::ref(task_queue), std::ref(task_mtx),
                                    std::ref(exit));
  }
  LOG_DEBUG("ntask %d, timestamp %f", n_tasks, timeMs());

  // let experiments start roughly same time
  FakeControlData ccc;
  LOG_IF_ERROR(::send(ctrl_fd, &ccc, sizeof(ccc), 0) != sizeof(ccc), "send control msg failed");
  LOG_IF_ERROR(::recv(ctrl_fd, &ccc, sizeof(ccc), MSG_WAITALL) != sizeof(ccc), "recv ccc confirm failed");

  for (int i = 0; i < N_EXP; ++i) {
    
    // LOG_DEBUG("send ccc");
    double s = timeMs();
    // launch tasks int to queue
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      for (int j = 0; j < n_tasks; ++j) {
        SocketTask* t = &tasks[j];
        t->stage = 1;
        t->ptr = (char*)buffer + j * SOCK_TASK_SIZE;
        t->size = SOCK_TASK_SIZE;
        t->exp_id = i;
        task_queue.push(t);
      }
    }

    double m1 = timeMs();
    LOG_DEBUG("launched send tasks %d", i);

    // wait for completion
    int n_complete = 0;
    while (n_complete != n_tasks){
      n_complete = 0;
      // double x = timeMs();
      for (int k = 0; k < n_tasks; ++k) {
        if (tasks[k].stage == 2) n_complete++;
      }
      // LOG_DEBUG("check cost %f ms, n_complete %d", timeMs() - x, n_complete);
    }

    double e = timeMs();
    LOG_INFO("send, exp %d, bw %f Gbps, size %d, time %f ms, launch cost %f ms", i, SOCK_REQ_SIZE * 8 / (e - s) / 1e6, SOCK_REQ_SIZE, (e-s), (m1 - s));

    for (int k = 0; k < n_tasks; ++k) {
      tasks[k].stage = 0;
    }

    LOG_IF_ERROR(::recv(ctrl_fd, &ccc, sizeof(ccc), MSG_WAITALL) != sizeof(ccc), "recv ccc confirm failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  exit = true;
  for (auto& t : background_threads) {
    t.join();
  }
}

void recvThread(int fd, std::queue<SocketTask*>& task_queue, std::mutex& mtx, bool& exit, 
      std::unordered_map<int, int>& completion_table, std::mutex& completion_mtx) {

  SocketTask* task;
  FakeControlData ctrl_msg;
  bool control_received = false;
  int ctrl2;
  int recv_count = 0;
  void* tmp_buff = malloc(SOCK_TASK_SIZE);

  while (!exit) {
    LOG_IF_ERROR(::recv(fd, &ctrl_msg, sizeof(ctrl_msg), MSG_WAITALL) !=
                      sizeof(ctrl_msg),
                  "receive control msg failed");
    if (ctrl_msg.exit == 1) {LOG_DEBUG("recv fd %d, recv count %d", fd, recv_count); return;}

    int ret = ::recv(fd, tmp_buff, SOCK_TASK_SIZE, MSG_WAITALL);
    LOG_IF_ERROR(ret != SOCK_TASK_SIZE, "error while recv data, ret %d", ret);
    {
      std::lock_guard<std::mutex> lk(completion_mtx);
      auto found = completion_table.find(ctrl_msg.exp_id);
      if (found == completion_table.end()) {
        completion_table[ctrl_msg.exp_id] = 1;
      } else {
        completion_table[ctrl_msg.exp_id] += 1;
      }
    }

    // if (!control_received) {
    //   // recv ctrl msg
    //   LOG_IF_ERROR(::recv(fd, &ctrl_msg, sizeof(ctrl_msg), MSG_WAITALL) !=
    //                    sizeof(ctrl_msg),
    //                "receive control msg failed");
    //   control_received = true;
    // }
    // if (ctrl_msg.exit == 1) {LOG_DEBUG("recv fd %d, recv count %d", fd, recv_count); return;}

    // while(task_queue.empty()) {
    //   std::this_thread::yield();
    // }

    // {
    //   std::lock_guard<std::mutex> lk(mtx);
    //   if (task_queue.empty()) {
    //     task = nullptr;
    //   } else {
    //     task = task_queue.front();
    //     task_queue.pop();
    //   }
    // }
    
    // if (task != nullptr) {
    //   double s = timeMs();
    //   int ret = ::recv(fd, task->ptr, task->size, MSG_WAITALL);
    //   LOG_IF_ERROR(ret != task->size, "error while recv data, ret %d", ret);

    //   // LOG_IF_ERROR(::send(fd, &ctrl2, sizeof(ctrl2), MSG_WAITALL) != sizeof(ctrl2), "fail sending confirmation");

    //   task->stage = 2;
    //   double e = timeMs();
    //   // LOG_DEBUG("recv fd %d, bw %f Gbps, ptr %p, size %d", fd, task->size / (e - s) / 1e6, task->ptr, task->size);
    //   task = nullptr;
    //   control_received = false;

    //   recv_count++;
    // }

  }

  
}

void clientMode(std::string& remote_ip, int remote_port) {
  int ip[4];
  ipStrToInts(remote_ip, ip);

  int ctrl_fd = createSocketClient(ip, remote_port, true);
  std::vector<int> data_fds;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    int fd = createSocketClient(ip, remote_port, false);
    data_fds.push_back(fd);
  }
  LOG_DEBUG("cli built data links to %s:%d", remote_ip.c_str(), remote_port);

  SocketTask tasks[MAX_TASKS];
  std::queue<SocketTask*> task_queue;
  std::mutex task_mtx;

  void* buffer = malloc(SOCK_REQ_SIZE);
  int n_tasks = SOCK_REQ_SIZE / SOCK_TASK_SIZE;

  std::unordered_map<int, int> completion_table;
  std::mutex completion_mtx;

  std::vector<std::thread> background_threads;
  bool exit = false;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    background_threads.emplace_back(recvThread, data_fds[i],
                                    std::ref(task_queue), std::ref(task_mtx),
                                    std::ref(exit), std::ref(completion_table), std::ref(completion_mtx));
  }

  // experiments, start from roughly same time
  FakeControlData ccc;
  LOG_IF_ERROR(::recv(ctrl_fd, &ccc, sizeof(ccc), MSG_WAITALL)!= sizeof(ccc), "fail recv ctrl msg");
  LOG_IF_ERROR(::send(ctrl_fd, &ccc, sizeof(ccc), MSG_WAITALL)!= sizeof(ccc), "fail send ctrl msg confirmation");


  for (int i = 0; i < N_EXP; ++i) {
    
    // LOG_DEBUG("recved ccc");
    double s = timeMs();
    // launch tasks int to queue
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      for (int j = 0; j < n_tasks; ++j) {
        SocketTask* t = &tasks[j];
        t->stage = 1;
        t->ptr = (char*)buffer + j * SOCK_TASK_SIZE;
        t->size = SOCK_TASK_SIZE;
        t->exp_id = i;
        task_queue.push(t);
      }
    }
    double m1 = timeMs();

    // wait for completion
    int n_complete = 0;
    auto found = completion_table.find(i);
    while (found == completion_table.end()) {
      found = completion_table.find(i);
    }
    while (completion_table[i] != n_tasks) {}

    double e = timeMs();

    // int match = memcmp(send_buff, buffer, SOCK_REQ_SIZE);
    // double recv_sum = floatSummary((float*)buffer, SOCK_REQ_SIZE / sizeof(float));
    // LOG_INFO("recv, exp %d, bw %f Gbps, size %d, time %f ms, launch cost %f ms, integrity %s, recv_sum %f, send_sum %f", i, SOCK_REQ_SIZE * 8 / (e - s) / 1e6,
    //          SOCK_REQ_SIZE, (e -s), (m1 - s), match == 0 ? "true":"false", recv_sum, floatSummary((float*)send_buff, SOCK_REQ_SIZE / sizeof(float)));

    // LOG_INFO("recv, exp %d, bw %f Gbps, size %d, time %f ms, launch cost %f ms, recv_sum %f", i, SOCK_REQ_SIZE * 8 / (e - s) / 1e6,
    //          SOCK_REQ_SIZE, (e -s), (m1 - s), recv_sum);
    LOG_INFO("recv, exp %d, bw %f Gbps, size %d, time %f ms, launch cost %f ms", i, SOCK_REQ_SIZE * 8 / (e - s) / 1e6,
             SOCK_REQ_SIZE, (e -s), (m1 - s));

    for (int k = 0; k < n_tasks; ++k) {
      tasks[k].stage = 0;
    }
    // memset(buffer, 0, SOCK_REQ_SIZE);
    LOG_IF_ERROR(::send(ctrl_fd, &ccc, sizeof(ccc), MSG_WAITALL)!= sizeof(ccc), "fail send ctrl msg confirmation");
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  exit = true;
  for (auto& t : background_threads) {
    t.join();
  }
}


int main(int argc, char* argv[]) {
  if (argc < 4) {
    LOG_INFO("require mode, remote-ip, port");
  }
  int mode = std::stoi(argv[1]);
  std::string remote_ip = std::string(argv[2]);
  int port = std::stoi(argv[3]);

  send_buff = malloc(SOCK_REQ_SIZE);
  memset(send_buff, 0, SOCK_REQ_SIZE);
  fillVals<float>((float*)send_buff, SOCK_REQ_SIZE / sizeof(float));
  LOG_DEBUG("send_buff summary %f", floatSummary((float*)send_buff, SOCK_REQ_SIZE / sizeof(float)));

  std::thread server(serverMode, port);
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::thread client(clientMode, std::ref(remote_ip), port);

  server.join();
  client.join();

  // if (mode > 0) {
  //   // listen server 
  //   serverMode(port);
  // } else {
  //   clientMode(remote_ip, port);
  // }
}