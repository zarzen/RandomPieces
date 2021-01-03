#ifndef SDCC_TCP_H
#define SDCC_TCP_H

#include <arpa/inet.h>
#include <cstring>
#include <memory>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace sdcc {
#define ERRNO_SUCCESS 0
#define ERRNO_TCP 1

#ifndef SO_EE_ORIGIN_ZEROCOPY
#define SO_EE_ORIGIN_ZEROCOPY 5
#endif

#ifndef SO_ZEROCOPY
#define SO_ZEROCOPY 60
#endif

#ifndef SO_EE_CODE_ZEROCOPY_COPIED
#define SO_EE_CODE_ZEROCOPY_COPIED 1
#endif

#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY 0x4000000
#endif

#define EXIT_ON_ERR(cond, exit_msg)                                            \
  do {                                                                         \
    if (cond) {                                                                \
      fprintf(stderr, "%s \n", exit_msg);                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

class TCPServer;
class TCPAgent;
class TCPClient;

class TCPServer {
public:
  TCPServer(std::string address, int port, int listen_num = 4,
            bool noDelay = false);
  ~TCPServer();
  std::shared_ptr<TCPAgent> acceptCli(bool noDelay = false);
  int getPort();

private:
  std::string _address;
  int _port;
  int _server_fd;
};

class TCPAgent {
protected:
  int _conn_fd;
  struct sockaddr_in _addr;

public:
  TCPAgent(int conn_fd);
  TCPAgent(int conn_fd, sockaddr_in addr);
  ~TCPAgent();

  int send(const char* data, size_t size);
  int send(const void* data, size_t size);
  int zsend(const char* data, size_t size);
  int recv(void* data, size_t size);
  int irecv(void* data, size_t size);
  int sendWithLength(const char* data, size_t size);
  int sendWithLength(const std::shared_ptr<char> data, size_t size);
  int recvWithLength(std::shared_ptr<char>& data, size_t& size);
  int sendString(const std::string data);
  int recvString(std::string& data);

  std::string getIP();
  int getPort();
  int getFd();
};

class TCPClient : public TCPAgent {
public:
  TCPClient(std::string address, int port, bool noDelay = false);
  ~TCPClient();

private:
  std::string _address;
  int _port;
};
} // namespace sdcc

#endif