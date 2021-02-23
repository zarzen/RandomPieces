#include "utils.h"
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sstream>
#include "common_structs.h"
#include "logger.h"

void initTaskInfo(hostDevShmInfo** info){
  hostAlloc<hostDevShmInfo>(info, 1);
  (*info)->tail = 0;
  (*info)->head = N_HOST_MEM_SLOTS;
  for (int i = 0; i < N_HOST_MEM_SLOTS; ++i) {
    void* pinned;
    CUDACHECK(cudaHostAlloc(&pinned, (MEM_SLOT_SIZE), cudaHostAllocMapped));
    (*info)->ptr_fifo[i] = pinned;
  }
}

void hostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
}

double timeMs() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count() /
         1e6;
};

uint64_t getHash(const char* string, int n) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

bool getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return false;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen-1)) i++;
  hostname[i] = '\0';
  return true;
}

#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
uint64_t getHostHash(void) {
  char hostHash[1024];
  char *hostId;

  // Fall back is the full hostname if something fails
  (void) getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  if ((hostId = getenv("NCCL_HOSTID")) != NULL) {
    // INFO(NCCL_ENV, "NCCL_HOSTID set by environment to %s", hostId);
    LOG_DEBUG("NCCL_HOSTID set by environment to %s", hostId);
    strncpy(hostHash, hostId, sizeof(hostHash));
  } else {
    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char *p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
      }
    }
    fclose(file);
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  LOG_DEBUG("unique hostname '%s'", hostHash);

  return getHash(hostHash, strlen(hostHash));
}

void ipStrToInts(std::string& ip, int* ret) {
  std::istringstream is(ip);
  std::string i;
  int idx = 0;
  while (std::getline(is, i, '.')){
    ret[idx] = std::stoi(i);
    ++idx;
    if (idx >= 4) return;
  }
}

bool createListenSocket(int* fd, int port) {
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  LOG_IF_ERROR(socket_fd == -1, "open socket error");

  int opt = 1;
  int ret;
  ret = setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt));
  LOG_IF_ERROR(ret == -1, "setsockopt failed");

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  ret = ::bind(socket_fd, (struct sockaddr*)&addr, sizeof(addr));
  LOG_IF_ERROR(ret < 0, "socket bind to port %d failed", port);

  LOG_IF_ERROR(listen(socket_fd, 16384) < 0, "socket listen failed");
  *fd = socket_fd;
  return true;
}

void getSocketPort(int* fd, int* port) {
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  LOG_IF_ERROR(getsockname(*fd, (struct sockaddr*)&sin, &len) == -1,
               "Err while getting port number.");
  *port = ntohs(sin.sin_port);
}

std::string getSocketIP(int& fd) {
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);

  LOG_IF_ERROR(getsockname(fd, (struct sockaddr*)&sin, &len) == -1,
              "getsockname failed %s", strerror(errno));

  struct in_addr ipAddr = sin.sin_addr;
  char str[INET_ADDRSTRLEN + 1] = {'\0'};
  inet_ntop(AF_INET, &ipAddr, str, INET_ADDRSTRLEN);
  return std::string(str, INET_ADDRSTRLEN);
}

// TODO: 
int socketAccept(int& server_fd, bool tcp_no_delay) {
  int cli_fd;
  struct sockaddr_in addr;
  auto addr_len = sizeof(addr);
  cli_fd = accept(server_fd, (struct sockaddr*)&addr, (socklen_t*)&addr_len);
  LOG_IF_ERROR(cli_fd < 0, "accept client failed, server fd %d", server_fd);

  if (tcp_no_delay) {
    int opt = 1;
    auto ret = setsockopt(cli_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    LOG_IF_ERROR(ret < 0, "enabling TCP_NODELAY failed");
  }
  return cli_fd;
}

std::string ipIntsToStr(int* p, int n) {
  std::stringstream ss;
  for (int i = 0; i < n; ++i) {
    ss << p[i];
    if (i != n-1) ss << ".";
  }
  return ss.str();
}

// TODO: 
int createSocketClient(int* ip, int port, bool no_delay) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  LOG_IF_ERROR(fd == 0, "create socket fd failed");

  if (no_delay) {
    int opt = 1;
    auto ret = setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    LOG_IF_ERROR(ret < 0, "enabling TCP_NODELAY failed");
  }

  struct sockaddr_in serv_addr;
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);
  std::string server_ip = ipIntsToStr(ip, 4);
  int ret = inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr);
  LOG_IF_ERROR(ret <= 0, "converting ip addr failed");

  bool retry = false;
  int MAX_RETRY = 10000;
  int retry_count = 0;
  do {
    int ret = ::connect(fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    if (ret != 0) retry = true;
    retry_count++;
    LOG_IF_ERROR((ret != 0 && retry_count % 1000 == 0), "connecting returned %s, retrying", strerror(errno));
  } while (retry && retry_count < MAX_RETRY);

  LOG_IF_ERROR(retry, "connect to %s:%d failed: %s", server_ip.c_str(), port, strerror(errno));
  return fd;
}