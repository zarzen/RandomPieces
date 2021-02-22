#pragma once

#include <cstdint>

struct peerInfo {
  int rank;
  int nranks;
  int dev_idx;
  int ip[4]; // for connection building
  int port; // for connection building
  uint64_t host_hash;
};

class RendezvousServer {

public:
  RendezvousServer(int port, int nranks);
};

class RendezvousClient {

public: 
  RendezvousClient(int* server_ip, int server_port);
  void registerPeer(peerInfo& info);
  peerInfo getPeerInfo(int peer);
};