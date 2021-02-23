#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <thread>
#include "common_structs.h"

typedef enum { push = 0, pull = 1 } RendezvousRequest_t;

struct RendezvousRequest {
  RendezvousRequest_t type;
  RankInfo info;
};

class RendezvousServer {
  int server_fd;
  int nranks;
  std::vector<int> cli_fds;
  bool accepted_all;
  std::unordered_map<int, RankInfo> rank_infos;
  bool shutdown;

  void waitAcceptAll();
  std::thread background_thd;
  static void persistenServiceThread(RendezvousServer* server);
  bool getRequest(int fd, RendezvousRequest* req_buff);
  bool handleRequest(RendezvousRequest& req, int cli_fd);

public:
  RendezvousServer(int port, int nranks);
  ~RendezvousServer();
};

class RendezvousClient {
  int fd;
public: 
  RendezvousClient(int* server_ip, int server_port);
  void registerRankInfo(RankInfo& info);
  RankInfo getPeerInfo(int peer);
};