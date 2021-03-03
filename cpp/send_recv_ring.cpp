#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "tcp.hpp"

std::string RAND_STR = "zxcvbnmlkjhgfdsaqwertyuiop0987654321";

void init_buf(char* buf, size_t buf_size) {
  for (size_t i = 0; i < buf_size; i++) {
    int idx = i % RAND_STR.length();
    *(buf + i) = RAND_STR.at(idx);
  }
}

void client_handle_thd(TCPAgent* tcpAgent,
                       int idx,
                       std::vector<size_t>* sizes) {
  // receive data from client
  std::shared_ptr<char> buf;
  size_t recv_size;
  while (true) {
    tcpAgent->recvWithLength(buf, recv_size);
    (*sizes)[idx] += recv_size;
  }
}

void tcp_server_thd(std::string ip, int port, int* num_conn, int target_conn) {
  TCPServer server(ip, port, target_conn);
  std::vector<std::thread> handlers;
  std::vector<size_t> sizes;
  std::vector<std::shared_ptr<TCPAgent>> tcp_conns;
  std::vector<size_t> pre_sizes;

  for (int i = 0; i < target_conn; i++) {
    std::shared_ptr<TCPAgent> cAgent = server.acceptCli();
    tcp_conns.push_back(cAgent);

    std::thread cth(client_handle_thd, tcp_conns[i].get(), i, &sizes);
    handlers.push_back(std::move(cth));
    sizes.push_back(0);

    (*num_conn) += 1;
    std::cout << "accepted tcp conn " << i << "\n";
    pre_sizes.push_back(0);
  }

  size_t pre_size = 0;
  
  while (true) {

    int interval = 500; // ms
    // std::this_thread::sleep_for(std::chrono::seconds(interval));
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    // size_t cur_size = 0UL;
    for (int i = 0; i < sizes.size(); i++) {
      // cur_size += sizes[i];
      size_t d = sizes[i] - pre_sizes[i];
      LOG_INFO("conn %d, bw %f Gbps", i, (d * 8 / float(interval)/1e6));
      pre_sizes[i] = sizes[i];
    }
  }
}

void msg_client_thd(std::string server_ip, int port, size_t msg_size) {
  // size_t buf_size = 10 * 1024 * 1024; // 10MB
  char* buf = new char[msg_size];
  init_buf(buf, msg_size);

  TCPClient client(server_ip, port);
  std::cout << "launched one tcp connect\n";

  while (true) {
    client.sendWithLength(buf, msg_size);
  }
};

int main(int argc, char* argv[]) {
  /* 0. bin-name
   * 1. local ip
   * 2. local port (for listening)
   * 3. next ip (for next hop)
   * 4. next port
   * 5. mode: 0, rank-0 or not
   * 6. num_tcp_conn
   * 7. msg size
   */
  if (argc < 8) {
    std::cerr << "args error\n";
    return -1;
  }

  std::string local_ip = std::string(argv[1]);
  int local_port = std::stoi(argv[2]);
  std::string next_ip = std::string(argv[3]);
  int next_port = std::stoi(argv[4]);
  int mode = std::stoi(argv[5]);
  int num_tcp = std::stoi(argv[6]);

  // default choice
  size_t msg_size = std::stoull(argv[7]);
  msg_size = msg_size / num_tcp;  // each TCP connection take part of the data
  std::cout << "send msg size" << msg_size << "\n";
  int num_clis = 0;
  std::thread local_server(tcp_server_thd, local_ip, local_port, &num_clis,
                           num_tcp);
  if (mode == 0) {
    // rank 0
    // don't need to wait for num_clis == num_tcp
    // start connecting to next hop
  } else {
    while ((num_clis) != num_tcp) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  std::vector<std::thread> clients;
  for (int i = 0; i < num_tcp; i++) {
    std::thread cthd(msg_client_thd, next_ip, next_port, msg_size);
    clients.push_back(std::move(cthd));
  }

  // need to join threads
  for (int i = 0; i < clients.size(); i++) {
    clients[i].join();
  }
}
