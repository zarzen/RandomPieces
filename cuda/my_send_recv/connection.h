#pragma once
#include <cstdlib>

typedef enum { Net = 0, P2P = 1, SHM = 2} ConnectionType_t;

// network, shm, p2p through this interface
class Connection {
 public:
  // only for CPU buffers
  virtual void sendCtrl(void* buff, size_t nbytes) = 0;
  // only for CPU buffers
  virtual void sendData(void* buff, size_t nbytes) = 0;

  virtual ConnectionType_t getType() = 0;
};

struct NetSendConnArgs {
  int* peer_ip;
  int peer_port;
  int n_socks;
  int n_threads;
};

struct NetRecvConnArgs {
  int ctrl_fd;
  int* data_fds;
  int n_socks; // num of data socks
  int n_threads;
};

// TCP network connection
class NetConnection : public Connection {
  int* data_fds;
  int ctrl_fd;
public:
 NetConnection(NetSendConnArgs& args);
 NetConnection(NetRecvConnArgs& args);

 void sendCtrl(void* buff, size_t nbytes);
 void sendData(void* buff, size_t nbytes);

 ConnectionType_t getType();
};

class P2PConnection : public Connection {

public:
  P2PConnection();
  void sendCtrl(void* buff, size_t nbytes);
  void sendData(void* buff, size_t nbytes);

  ConnectionType_t getType();
};
