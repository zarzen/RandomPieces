
#include "tcp.hpp"
#include <asm-generic/socket.h>
#include <linux/errqueue.h>
#include <linux/if_packet.h>
#include <sys/socket.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include "error.h"
#include "logger.h"
#include <errno.h>

using namespace std;

TCPServer::TCPServer(string address, int port, int listen_num, bool noDelay)
    : _address(address), _port(port) {
  _server_fd = socket(AF_INET, SOCK_STREAM, 0);
  EXIT_ON_ERR(_server_fd == 0, "util/common/TCPServer CreateSocket");

  int opt = 1;
  EXIT_ON_ERR(setsockopt(_server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                         &opt, sizeof(opt)),
              "util/common/TCPServer SetSocketOption");
  if (noDelay) {
    LOG_DEBUG("enabled TCP_NODELAY");
    int yes = 1;
    EXIT_ON_ERR(
        setsockopt(_server_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)),
        "util/common/TCPServer SetSocketOption");
  }

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(_port);
  EXIT_ON_ERR(inet_pton(AF_INET, _address.c_str(), &addr.sin_addr) <= 0,
              "util/common/TCPServer ConvertStringAddress");
  EXIT_ON_ERR(::bind(_server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0,
              "util/common/TCPServer BindPort");

  EXIT_ON_ERR(listen(_server_fd, listen_num) < 0,
              "util/common/TCPServer Listen");
}

int TCPServer::getPort() {
  // if port is 0 at first system will assign randomly
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  EXIT_ON_ERR(getsockname(_server_fd, (struct sockaddr*)&sin, &len) == -1,
              "Err while getting port number.");
  _port = ntohs(sin.sin_port);
  return _port;
}

TCPServer::~TCPServer() {
  close(_server_fd);
}

shared_ptr<TCPAgent> TCPServer::acceptCli(bool noDelay) {
  int conn_fd;
  struct sockaddr_in addr;
  size_t addrlen = sizeof(addr);
  conn_fd = accept(_server_fd, (struct sockaddr*)&addr, (socklen_t*)&addrlen);
  if (conn_fd < 0) {
    perror("util/common/TCPServer AcceptConnection");
    exit(EXIT_FAILURE);
  }

  if (noDelay) {
    LOG_DEBUG("enabled TCP_NODELAY");
    int yes = 1;
    EXIT_ON_ERR(
        setsockopt(conn_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)),
        "util/common/TCPServer SetSocketOption");
  }

  /* Set a fixed size of sndbuf/rcvbuf (8MB) */
  if (const char* env_p = std::getenv("TRANS_TCP_BUF")) {
    int opt = std::stoi(env_p);
    EXIT_ON_ERR(
        setsockopt(conn_fd, SOL_SOCKET, SO_SNDBUF, (char*)&opt, sizeof(opt)),
        "failed to set SO_SNDBUF sock opt");
    EXIT_ON_ERR(
        setsockopt(conn_fd, SOL_SOCKET, SO_RCVBUF, (char*)&opt, sizeof(opt)),
        "failed to set SO_RCVBUF sock opt");
  }

  if (const char* env_p = std::getenv("TRANS_TCP_SYN")) {
    /* Single syn retry */
    int opt = std::stoi(env_p);
    if (opt > 0) {
      EXIT_ON_ERR(setsockopt(conn_fd, IPPROTO_TCP, TCP_SYNCNT, (char*)&opt,
                             sizeof(opt)),
                  "failed to set TCP_SYNCNT sock opt");
    }
  }

  return shared_ptr<TCPAgent>(new TCPAgent(conn_fd, addr));
}

TCPAgent::TCPAgent(int conn_fd) : _conn_fd(conn_fd) {}
TCPAgent::TCPAgent(int conn_fd, struct sockaddr_in addr)
    : _conn_fd(conn_fd), _addr(addr) {}

std::string TCPAgent::getIP() {
  struct in_addr ipAddr = this->_addr.sin_addr;
  char str[INET_ADDRSTRLEN + 1] = {'\0'};
  inet_ntop(AF_INET, &ipAddr, str, INET_ADDRSTRLEN);
  return std::string(str, INET_ADDRSTRLEN);
}

TCPAgent::~TCPAgent() {
  shutdown(_conn_fd, SHUT_RDWR);
  close(_conn_fd);
}

static bool do_recv_completion(int fd) {
  struct sock_extended_err* serr;
  struct msghdr msg = {};
  struct cmsghdr* cm;
  uint32_t hi, lo, range;
  int ret, zerocopy;
  char control[100];

  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);

  ret = recvmsg(fd, &msg, MSG_ERRQUEUE);
  if (ret == -1 && errno == EAGAIN)
    return false;
  if (ret == -1)
    error(1, errno, "recvmsg notification");
  if (msg.msg_flags & MSG_CTRUNC)
    error(1, errno, "recvmsg notification: truncated");

  // cm = CMSG_FIRSTHDR(&msg);
  // if (!cm)
  //   error(1, 0, "cmsg: no cmsg");
  // if (!((cm->cmsg_level == SOL_IP && cm->cmsg_type == IP_RECVERR) ||
  //       (cm->cmsg_level == SOL_IPV6 && cm->cmsg_type == IPV6_RECVERR) ||
  //       (cm->cmsg_level == SOL_PACKET && cm->cmsg_type == PACKET_TX_TIMESTAMP)))
  //   error(1, 0, "serr: wrong type: %d.%d", cm->cmsg_level, cm->cmsg_type);

  // serr = (struct sock_extended_err*)CMSG_DATA(cm);

  // if (serr->ee_origin != SO_EE_ORIGIN_ZEROCOPY)
  //   error(1, 0, "serr: wrong origin: %u", serr->ee_origin);
  // if (serr->ee_errno != 0)
  //   error(1, 0, "serr: wrong error code: %u", serr->ee_errno);

  // hi = serr->ee_data;
  // lo = serr->ee_info;
  // range = hi - lo + 1;

  /* Detect notification gaps. These should not happen often, if at all.
   * Gaps can occur due to drops, reordering and retransmissions.
   */
  // if (lo != next_completion)
  //   fprintf(stderr, "gap: %u..%u does not append to %u\n", lo, hi,
  //           next_completion);
  // next_completion = hi + 1;

  // zerocopy = !(serr->ee_code & SO_EE_CODE_ZEROCOPY_COPIED);
  // if (zerocopied == -1)
  //   zerocopied = zerocopy;
  // else if (zerocopied != zerocopy) {
  //   fprintf(stderr, "serr: inconsistent\n");
  //   zerocopied = zerocopy;
  // }

  // if (cfg_verbose >= 2)
  //   fprintf(stderr, "completed: %u (h=%u l=%u)\n", range, hi, lo);

  // completions += range;
  return true;
}

int TCPAgent::zsend(const char* data, size_t size) {
  int ret = ::send(_conn_fd, data, size, MSG_ZEROCOPY);
  // TODO not sure whether the MSG_ZEROCOPY flag will result
  // asyn operation
  // and typically the zero copy work with asyn socket operation
  // which require the completion notification handling.
  if (ret != size) {
    LOG_ERROR("zero copy send ret %d", ret);
    return ERRNO_TCP;
  }
  while(!do_recv_completion(_conn_fd)){
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return ERRNO_SUCCESS;
}

int TCPAgent::send(const char* data, size_t size) {
  auto ret = ::send(_conn_fd, data, size, 0);
  if (ret != size) {
    LOG_ERROR("tcp send failed, return %zu, size %zu, error msg %s", ret, size,
              strerror(errno));
    return ERRNO_TCP;
  }
  return ERRNO_SUCCESS;
}

int TCPAgent::recv(char* data, size_t size) {
  auto ret = ::recv(_conn_fd, data, size, MSG_WAITALL);
  if (ret != size) {
    LOG_ERROR("tcp recv failed, return %zu, size %zu, error msg %s", ret, size,
              strerror(errno));
    return ERRNO_TCP;
  }
  return ERRNO_SUCCESS;
}

int TCPAgent::sendWithLength(const char* data, size_t size) {
  int status;
  status = send((char*)&size, sizeof(size));
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  status = send(data, size);
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  return ERRNO_SUCCESS;
}

int TCPAgent::sendWithLength(const shared_ptr<char> data, size_t size) {
  int status;
  status = send((char*)&size, sizeof(size));
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  status = send(data.get(), size);
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  return ERRNO_SUCCESS;
}

int TCPAgent::recvWithLength(shared_ptr<char>& data, size_t& size) {
  // cout << "tcpRecvWithLength (1): " << sizeof(size) << endl;
  int status;

  status = recv((char*)&size, sizeof(size));
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  if (size > 10000000)
    return ERRNO_TCP;
  data.reset((char*)malloc(size));
  status = recv(data.get(), size);
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;

  return ERRNO_SUCCESS;
}

int TCPAgent::sendString(const string data) {
  int status;
  size_t data_size = data.size();
  status = send((char*)&data_size, sizeof(data_size));
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  status = send(data.c_str(), data_size);
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  return ERRNO_SUCCESS;
}

int TCPAgent::recvString(string& data) {
  int status;

  size_t data_size = 0;
  status = recv((char*)&data_size, sizeof(data_size));
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;
  if (data_size > 1000)
    return ERRNO_TCP;

  char* buffer = (char*)malloc(data_size);
  memset(buffer, 0, data_size);
  status = recv(buffer, data_size);
  if (status != ERRNO_SUCCESS)
    return ERRNO_TCP;

  data = string(buffer, data_size);
  free(buffer);
  return ERRNO_SUCCESS;
}

TCPClient::TCPClient(string address, int port, bool noDelay)
    : TCPAgent(0), _address(address), _port(port) {
  _conn_fd = socket(AF_INET, SOCK_STREAM, 0);
  EXIT_ON_ERR(_conn_fd == 0, "util/common/TCPClient CreateSocket");

  if (noDelay) {
    LOG_DEBUG("enabled TCP_NODELAY");
    int yes = 1;
    EXIT_ON_ERR(
        setsockopt(_conn_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)),
        "util/common/TCPServer SetSocketOption");
  }
  /* Set a fixed size of sndbuf/rcvbuf (8MB) */
  if (const char* env_p = std::getenv("TRANS_TCP_BUF")) {
    int opt = std::stoi(env_p);
    EXIT_ON_ERR(
        setsockopt(_conn_fd, SOL_SOCKET, SO_SNDBUF, (char*)&opt, sizeof(opt)),
        "failed to set SO_SNDBUF sock opt");
    EXIT_ON_ERR(
        setsockopt(_conn_fd, SOL_SOCKET, SO_RCVBUF, (char*)&opt, sizeof(opt)),
        "failed to set SO_RCVBUF sock opt");
  }

  if (const char* env_p = std::getenv("TRANS_TCP_SYN")) {
    /* Single syn retry */
    int opt = std::stoi(env_p);
    if (opt > 0) {
      EXIT_ON_ERR(setsockopt(_conn_fd, IPPROTO_TCP, TCP_SYNCNT, (char*)&opt,
                             sizeof(opt)),
                  "failed to set TCP_SYNCNT sock opt");
    }
  }

  if (const char* env_p = std::getenv("TRANS_ZP")) {
    // set the zero copy flag for the socket
    int opt = std::stoi(env_p);
    if (opt > 0) {
      LOG_INFO("enable zero-copy at client");
      int val = 1;
      EXIT_ON_ERR(
          setsockopt(_conn_fd, SOL_SOCKET, SO_ZEROCOPY, &val, sizeof(val)),
          "fail to set SO_ZEROCOPY");
    }
  }

  struct sockaddr_in serv_addr;
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(_port);
  EXIT_ON_ERR(inet_pton(AF_INET, _address.c_str(), &serv_addr.sin_addr) <= 0,
              "util/common/TCPClient ConvertStringAddress");
  int repeat = 0;
  while (connect(_conn_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    LOG_DEBUG("error while connecting to %s:%d, try in 50ms", _address.c_str(),
                 _port);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    repeat++;
  }
}

int TCPAgent::getPort() {
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  EXIT_ON_ERR(getsockname(_conn_fd, (struct sockaddr*)&sin, &len) == -1,
              "Err while getting port number");
  return ntohs(sin.sin_port);
}

int TCPAgent::getFd() {
  return _conn_fd;
}

TCPClient::~TCPClient() {}
