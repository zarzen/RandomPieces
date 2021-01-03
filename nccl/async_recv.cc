#include "sdcc_tcp.h"
#include <memory>
#include <thread>
#include <iostream>

int listen_port = 5555;

void serverAcceptCli() {

  std::shared_ptr<sdcc::TCPServer> server =
      std::make_shared<sdcc::TCPServer>("0.0.0.0", listen_port, 4, true);
  std::shared_ptr<sdcc::TCPAgent> cli = server->acceptCli(true);
  char msg[132];
  cli->send(msg, sizeof(msg));
  while(true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

int main() {
  std::thread local_thd = std::thread(serverAcceptCli);

  std::shared_ptr<sdcc::TCPClient> cli =
      std::make_shared<sdcc::TCPClient>("172.31.88.106", listen_port, true);

  char msg[132];
  while (true) {
    
    int bytes = cli->irecv(msg, sizeof(msg));
    // int bytes = ::recv(cli->getFd(), msg, sizeof(msg), MSG_DONTWAIT);
    // int bytes = cli->recv(msg, sizeof(msg));
    if (bytes > 0) {
      if (bytes < sizeof(msg)) {
        cli->recv(msg+bytes, sizeof(msg) - bytes);
      }
      std::cout << "completed receive\n" << bytes << "\n";
    }
  }
}