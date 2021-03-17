#pragma once

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

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))


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