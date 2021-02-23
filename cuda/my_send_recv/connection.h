#pragma once
#include <cstdlib>
#include <vector>
#include "common_structs.h"

typedef enum { Net = 0, P2P = 1, SHM = 2} ConnectionType_t;

// network, shm, p2p through this interface
class Connection {
 public:
  // for GPU buffers, launch different kernels
  virtual void send(void* buff, size_t nbytes) = 0;
  // for GPU buffers
  virtual void recv(void* buff, size_t nbytes) = 0;

  virtual ConnectionType_t getType() = 0;
};

struct NetSendConnArgs {
  int self_rank;
  int peer_ip[4];
  int peer_port;
  int n_socks;
  int n_threads;
};

struct NetRecvConnArgs {
  int self_rank;
  int ctrl_fd;
  int n_socks; // num of data socks
  int n_threads;
};

struct PeerConnectionInfo {
  int peer_rank;
  ConnectionType_t conn_type;
};

// TCP network connection
class NetConnection : public Connection {
  std::vector<int> data_fds;
  int ctrl_fd;
  int is_send;
  int dev_idx;
  hostDevShmInfo* ctrl_buff;

  int listen_fd; // only used for recv connection
  void initBuffer();
public:
 NetConnection(NetSendConnArgs& args, int dev_idx);
 NetConnection(NetRecvConnArgs& args, int dev_idx);

 void send(void* buff, size_t nbytes);
 void recv(void* buff, size_t nbytes);

 ConnectionType_t getType();
};

class P2PConnection : public Connection {

public:
  P2PConnection();
  void send(void* buff, size_t nbytes);
  void recv(void* buff, size_t nbytes);

  ConnectionType_t getType();
};
