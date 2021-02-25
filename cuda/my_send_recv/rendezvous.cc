#include "rendezvous.h"
#include "utils.h"
#include "logger.h"
#include <sys/socket.h>
#include <queue>
#include <utility>

void RendezvousServer::waitAcceptAll() {
  // accept for all clients
  if (!accepted_all) {
    for (int i = 0; i < nranks; ++i) {
      int cli = socketAccept(server_fd, true);
      LOG_IF_ERROR(cli < 0, "accept client socket failed");
      cli_fds.push_back(cli);
      // LOG_DEBUG("rendezvous-server accept one client");
    }
    
    int echo_msg = 1;
    for (auto fd : cli_fds) {
      auto ret = ::send(fd, &echo_msg, sizeof(int), 0);
      LOG_IF_ERROR(ret != sizeof(int), "sending connection confirm msg failed to %s:", getSocketIP(fd).c_str());
    }
    accepted_all = true;
  }
}

bool RendezvousServer::getRequest(int fd, RendezvousRequest* req_buff) {
  int bytes = 0;
  int req_size = sizeof(RendezvousRequest);
  char* ptr = (char*)req_buff;
  bytes = ::recv(fd, ptr, req_size, MSG_DONTWAIT);
  if (bytes > 0) {
    if (bytes < req_size) {
      // LOG_DEBUG("getting remain data of a request of fd %d, remain %d", fd, req_size - bytes);
      int add_bytes = ::recv(fd, ptr + bytes, req_size - bytes, MSG_WAITALL);
    }
    return true;
  } else {
    return false;
  }
}

bool RendezvousServer::handleRequest(RendezvousRequest& req, int cli_fd) {
  if (req.type == push) {
    rank_infos[req.info.rank] = req.info;
    // LOG_DEBUG(
    //     "rendezvous-server registered rank info: rank %d, nranks %d, hosthash "
    //     "%lu, port %d",
    //     req.info.rank, req.info.nranks, req.info.host_hash, req.info.port);
    return true;
  }

  if (req.type == pull) {
    auto found = rank_infos.find(req.info.rank);
    if (found != rank_infos.end()) {
      // found, then send the info back to client
      int bytes = ::send(cli_fd, &(found->second), sizeof(found->second), 0);
      LOG_IF_ERROR(bytes != sizeof(RankInfo), "sending RankInfo back failed");
      LOG_DEBUG("rendez-server fulfilled pull request for rank %d", req.info.rank);
      return true;
    } else {
      // not available now
      // LOG_DEBUG("rendez-server unfulfilled for rank %d, data not available",
      //           req.info.rank);
      return false;
    }
  }
}

void RendezvousServer::persistenServiceThread(RendezvousServer* server) {
  int nranks = server->nranks;
  server->waitAcceptAll();

  // async serve all clients
  std::queue<std::pair<int, RendezvousRequest>> requests;
  while (!server->shutdown) {
    // for each cli, it try to receive request from client
    bool got_req = false;
    for (int i = 0; i < nranks; ++i) {
      int fd = server->cli_fds[i];
      RendezvousRequest req;
      got_req = server->getRequest(fd, &req);
      if (got_req) {
        requests.push(std::make_pair(fd, req));
        // LOG_DEBUG("rendez-server got request: fd %d, req-type %s, rank %d", fd,
        //           req.type == pull ? "pull" : "push", req.info.rank);
      }
    }

    std::queue<std::pair<int, RendezvousRequest>> unfulfilled_requests;
    while (!requests.empty()) {

      std::pair<int, RendezvousRequest> req_pair = requests.front();
      requests.pop();
      bool handled = server->handleRequest(req_pair.second, req_pair.first);
      if (!handled) unfulfilled_requests.push(req_pair);
      // LOG_DEBUG("rendez-server there are %lu requests unfulfilled",
      //           unfulfilled_requests.size());
    }
    requests.swap(unfulfilled_requests);
  }
  LOG_DEBUG("rendez-server persistent thread exits");
}

// TODO:
RendezvousServer::RendezvousServer(int port, int nranks): nranks(nranks), accepted_all(false), shutdown(false) {
  bool res = createListenSocket(&server_fd, port);
  LOG_IF_ERROR(!res, "creating listen socket for rendezvous server failed");

  background_thd = std::thread(persistenServiceThread, this);
}

RendezvousServer::~RendezvousServer() {
  shutdown = true;
  if (background_thd.joinable()) background_thd.join();
}


RendezvousClient::RendezvousClient(int* server_ip, int server_port) {
  fd = createSocketClient(server_ip, server_port, true);
  // wait for echo back from server
  int echo_msg;
  int bytes = ::recv(fd, &echo_msg, sizeof(echo_msg), 0);
  LOG_IF_ERROR(bytes != sizeof(echo_msg), "rendezvous recv echo from server failed");
}

void RendezvousClient::registerRankInfo(RankInfo& info) {
  RendezvousRequest req;
  req.type = push;
  req.info = info;
  int bytes = ::send(fd, &req, sizeof(req), 0);
  LOG_IF_ERROR(bytes != sizeof(req), "rendezvous-cli registering rank information failed");
}

// TODO:
RankInfo RendezvousClient::getPeerInfo(int peer) {
  RendezvousRequest req;
  req.type = pull;
  req.info.rank = peer;

  int bytes = ::send(fd, &req, sizeof(req), 0);
  LOG_IF_ERROR(bytes != sizeof(req), "rendezvous-cli pulling-send-req peer info failed");

  bytes = ::recv(fd, &req.info, sizeof(RankInfo), 0);
  LOG_IF_ERROR(bytes != sizeof(RankInfo), "rendezvous-cli pull-recv-info failed");
  return req.info;
}

