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


double timeMs() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count() /
         1e6;
};

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
#define SOCK_REQ_SIZE (2*1024 * 1024) // 512kB or 1MB
#define SOCK_TASK_SIZE (64 * 1024) // 64kB
// #define N_SOCK_REQ 4 // 4 slots 
#define MAX_TASKS (2 * 1024) // for test only

struct SocketTask {
  void* ptr;
  int size;
  int stage;
};

struct FakeControlData {
  int exit;
  char pad[16];
};


void sendThread(int fd, std::queue<SocketTask*>& task_queue, std::mutex& mtx, bool& exit) {

  SocketTask* task = nullptr;
  FakeControlData ctrl_msg;

  FakeControlData ctrl2;

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
        LOG_IF_ERROR(
            ::send(fd, &ctrl_msg, sizeof(ctrl_msg), MSG_WAITALL) != sizeof(ctrl_msg),
            "send ctrl msg error");

        double s = timeMs();
        int ret = ::send(fd, task->ptr, task->size, MSG_WAITALL);
        LOG_IF_ERROR(ret != task->size, "send data failed");

        LOG_IF_ERROR(::recv(fd, &ctrl2, sizeof(ctrl2), MSG_WAITALL) != sizeof(ctrl2), "failed at recv confirmation");

        task->stage = 2;
        double e = timeMs();
        LOG_DEBUG("fd %d, size %d, bw %f Gbps", fd, task->size, task->size * 8 / (e - s) / 1e6);

        task = nullptr;
        
      }
    }
  }
  ctrl_msg.exit = 1;
  LOG_IF_ERROR(::send(fd, &ctrl_msg, sizeof(ctrl_msg), 0) != sizeof(ctrl_msg),
               "send ctrl msg error");
}

#define N_EXP 100

void serverMode(int port) {
  int listen_fd;
  bool ret = createListenSocket(&listen_fd, port);
  LOG_IF_ERROR(ret == false, "create listen server failed");

  int ctrl_fd = socketAccept(listen_fd, true);

  std::vector<int> data_fds;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    int fd = socketAccept(listen_fd, true);
    data_fds.push_back(fd);
  }

  // experiments send message to client in producer consumer 
  SocketTask tasks[MAX_TASKS];
  std::queue<SocketTask*> task_queue;
  std::mutex task_mtx;

  void* buffer = malloc(SOCK_REQ_SIZE);
  int n_tasks = SOCK_REQ_SIZE / SOCK_TASK_SIZE;

  std::vector<std::thread> background_threads;
  bool exit = false;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    background_threads.emplace_back(sendThread, data_fds[i],
                                    std::ref(task_queue), std::ref(task_mtx),
                                    std::ref(exit));
  }
  LOG_DEBUG("ntask %d, timestamp %f", n_tasks, timeMs());
  FakeControlData ccc;
  for (int i = 0; i < N_EXP; ++i) {
    LOG_IF_ERROR(::send(ctrl_fd, &ccc, sizeof(ccc), 0) != sizeof(ccc), "send control msg failed");

    double s = timeMs();
    // launch tasks int to queue
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      for (int j = 0; j < n_tasks; ++j) {
        SocketTask* t = &tasks[j];
        t->stage = 1;
        t->ptr = (char*)buffer + j * SOCK_TASK_SIZE;
        t->size = SOCK_TASK_SIZE;
        task_queue.push(t);
      }
    }

    double m1 = timeMs();

    // wait for completion
    int n_complete = 0;
    while (n_complete != n_tasks){
      n_complete = 0;
      double x = timeMs();
      for (int i = 0; i < n_tasks; ++i) {
        if (tasks[i].stage == 2) n_complete++;
      }
      LOG_DEBUG("check cost %f ms, n_complete %d", timeMs() - x, n_complete);
    }

    double e = timeMs();
    LOG_INFO("send, exp %d, bw %f Gbps, size %d, time %f ms, launch cost %f ms", i, SOCK_REQ_SIZE * 8 / (e - s) / 1e6, SOCK_REQ_SIZE, (e-s), (m1 - s));

    for (int i = 0; i < n_tasks; ++i) {
      tasks[i].stage = 0;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  exit = true;
  for (auto& t : background_threads) {
    t.join();
  }
}

void recvThread(int fd, std::queue<SocketTask*>& task_queue, std::mutex& mtx, bool& exit) {
  SocketTask* task;
  FakeControlData ctrl_msg;
  bool control_received = false;
  int ctrl2;

  while (!exit) {
    if (!control_received) {
      // recv ctrl msg
      LOG_IF_ERROR(::recv(fd, &ctrl_msg, sizeof(ctrl_msg), MSG_WAITALL) !=
                       sizeof(ctrl_msg),
                   "receive control msg failed");
      control_received = true;
    }
    if (ctrl_msg.exit == 1) return;

    while(task_queue.empty()) {
      std::this_thread::yield();
    }

    {
      std::lock_guard<std::mutex> lk(mtx);
      if (task_queue.empty()) {
        task = nullptr;
      } else {
        task = task_queue.front();
        task_queue.pop();
      }
    }
    
    if (task != nullptr) {
      int ret = ::recv(fd, task->ptr, task->size, MSG_WAITALL);
      LOG_IF_ERROR(ret != task->size, "error while recv data, ret %d", ret);

      // LOG_IF_ERROR(::send(fd, &ctrl2, sizeof(ctrl2), MSG_WAITALL) != sizeof(ctrl2), "fail sending confirmation");

      task->stage = 2;
      task = nullptr;
      control_received = false;
    }

  }
}

void clientMode(std::string& remote_ip, int remote_port) {
  int ip[4];
  ipStrToInts(remote_ip, ip);

  int ctrl_fd = createSocketClient(ip, remote_port, true);
  std::vector<int> data_fds;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    int fd = createSocketClient(ip, remote_port, true);
    data_fds.push_back(fd);
  }

  SocketTask tasks[MAX_TASKS];
  std::queue<SocketTask*> task_queue;
  std::mutex task_mtx;

  void* buffer = malloc(SOCK_REQ_SIZE);
  int n_tasks = SOCK_REQ_SIZE / SOCK_TASK_SIZE;

  std::vector<std::thread> background_threads;
  bool exit = false;
  for (int i = 0; i < N_DATA_SOCK; ++i) {
    background_threads.emplace_back(recvThread, data_fds[i],
                                    std::ref(task_queue), std::ref(task_mtx),
                                    std::ref(exit));
  }

  // experiments
  FakeControlData ccc;
  for (int i = 0; i < N_EXP; ++i) {
    LOG_IF_ERROR(::recv(ctrl_fd, &ccc, sizeof(ccc), MSG_WAITALL)!= sizeof(ccc), "fail recv ctrl msg");

    double s = timeMs();
    // launch tasks int to queue
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      for (int j = 0; j < n_tasks; ++j) {
        SocketTask* t = &tasks[j];
        t->stage = 1;
        t->ptr = (char*)buffer + j * SOCK_TASK_SIZE;
        t->size = SOCK_TASK_SIZE;
        task_queue.push(t);
      }
    } 
    double m1 = timeMs();

    // wait for completion
    int n_complete = 0;
    while (n_complete != n_tasks) {
      n_complete = 0;
      for (int i = 0; i < n_tasks; ++i) {
        if (tasks[i].stage == 2)
          n_complete++;
      }
    }

    double e = timeMs();
    LOG_INFO("recv, exp %d, bw %f Gbps, size %d, time %f ms, launch cost %f ms", i, SOCK_REQ_SIZE * 8 / (e - s) / 1e6,
             SOCK_REQ_SIZE, (e -s), (m1 - s));

    for (int i = 0; i < n_tasks; ++i) {
      tasks[i].stage = 0;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
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