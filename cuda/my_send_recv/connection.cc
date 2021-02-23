#include "connection.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include <sys/socket.h>
#include "logger.h"

ConnectionType_t NetConnection::getType() {
  return Net;
}

void NetConnection::initBuffer() {
  cudaSetDevice(dev_idx);
  hostAlloc<hostDevShmInfo>(&ctrl_buff, 1);

}

static inline void sendNetConnectionInfo(int& rank, ConnectionType_t type, int& fd) {
  PeerConnectionInfo my_info;
  my_info.peer_rank = rank;
  my_info.conn_type = type;
  auto ret = ::send(fd, &my_info, sizeof(PeerConnectionInfo), 0);
  LOG_IF_ERROR(ret != sizeof(PeerConnectionInfo), "send connection info to peer failed");
}

int recvNewListenPort(int& fd) {
  int port;
  auto ret = ::recv(fd, &port, sizeof(int), 0);
  LOG_IF_ERROR(ret != sizeof(int), "recv new listen port failed");
  return port;
}

static inline void buildNetSendDataConns(int* ip, int port, int n, std::vector<int>& data_fds) {
  for (int i = 0; i < n; ++i) {
    int fd = createSocketClient(ip, port, true);
    LOG_IF_ERROR(fd < 0, "create net send data connection failed");
    data_fds.push_back(fd);
  }
}

// TODO: build connection with remote peer, launch n_threads with n_socks as data socks
NetConnection::NetConnection(NetSendConnArgs& args, int dev_idx): is_send(true), dev_idx(dev_idx) {
  initBuffer();
  ctrl_fd = createListenSocket(args.peer_ip, args.peer_port);
  sendNetConnectionInfo(args.self_rank, Net, ctrl_fd);
  int new_port = recvNewListenPort(ctrl_fd);
  data_fds.reserve(args.n_socks);
  buildNetSendDataConns(args.peer_ip, new_port, args.n_socks, this->data_fds);

  // TODO: launch threads for socket send tasks
}

static inline void sendNewListenPort(int& ctrl_fd, int& new_listen_fd) {
  int new_listen_port;
  getSocketPort(&new_listen_fd, &new_listen_port);
  auto ret = ::send(ctrl_fd, &new_listen_port, sizeof(int), 0);
  LOG_IF_ERROR(ret != sizeof(int), "send new listen port failed");
}

static inline void buildNetRecvDataConns(int& listen_fd, int& n, std::vector<int>& data_fds) {
  for (int i = 0; i < n; ++i) {
    int fd = socketAccept(listen_fd, true);
    LOG_IF_ERROR(fd < 0, "create net data recv connection failed");
    data_fds.push_back(fd);
  }
}

// TODO: start a local listen socket, send the port back, accept n_socks as data recv socks, and launch n_threads
NetConnection::NetConnection(NetRecvConnArgs& args, int dev_idx): is_send(false), dev_idx(dev_idx) {
  initBuffer();
  ctrl_fd = args.ctrl_fd;
  createListenSocket(&listen_fd, 0);
  sendNewListenPort(ctrl_fd, listen_fd);
  buildNetRecvDataConns(listen_fd, args.n_socks, this->data_fds);

  // TODO: launch threads for socket tasks
}

void NetConnection::send(void* buff, size_t count) {}
void NetConnection::recv(void* buff, size_t count) {}
