#include <cstdlib>
#include <iostream>
#include <string>
#include "sendrecv.h"
#include "utils.h"
#include "logger.h"

#define N_ELEM (1024 * 1024)
#define RAND_SEED 123
#define REPEAT_EXP 1

void fillRandFloats(float* buffer, int nelem) {
  srand(RAND_SEED);
  for (int i = 0; i < nelem; ++i) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    buffer[i] = r;
  }
}

int initBuffers(int argc,
                 char* argv[],
                 void** send_buff,
                 void** recv_buff,
                 void** host_buff,
                 void** host_tmp) {
  int nelem = N_ELEM;
  if (argc > 7) {
    nelem = std::stoi(argv[7]);
  }
  int nbytes = nelem * sizeof(float);
  *host_buff = malloc(nbytes);
  *host_tmp = malloc(nbytes);
  CUDACHECK(cudaMalloc(send_buff, nbytes));
  CUDACHECK(cudaMemset(*send_buff, 0, nbytes));
  CUDACHECK(cudaMalloc(recv_buff, nbytes));
  CUDACHECK(cudaMemset(*recv_buff, 0, nbytes));

  // init send buffer
  fillRandFloats((float*)*host_buff, nelem);
  cudaMemcpy(*send_buff, *host_buff, nbytes, cudaMemcpyDefault);

  return nelem;
}

CommunicatorArgs getArgs(int& argc, char* argv[]) {
  if (argc < 7) {
    printf(
        "require at least 6 arguments for benchmark (rendezvous_ip, "
        "rendezvous_port, rank, nranks, dev_idx, local_ip) \n");
  }
  CommunicatorArgs c;
  std::string rendez_ip_str = std::string(argv[1]);
  ipStrToInts(rendez_ip_str, c.rendezvous_ip);
  c.rendezvous_port = std::stoi(argv[2]);
  c.rank = std::stoi(argv[3]);
  c.nranks = std::stoi(argv[4]);
  c.dev_idx = std::stoi(argv[5]);
  std::string local_ip = std::string(argv[6]);
  ipStrToInts(local_ip, c.local_ip);
  c.send_stream = NULL;
  c.recv_stream = NULL;
  return c;
}

bool dataIntegrityCheck(void* dev_recv_buff, void* ref_buff, void* tmp_buff, int nelem);

int main(int argc, char* argv[]) {
  CommunicatorArgs context = getArgs(argc, argv);
  Communicator comm(context);
  // 
  cudaSetDevice(context.dev_idx);

  void *host_buff, *dev_send_buff, *dev_recv_buff, *host_tmp;
  int nelem = initBuffers(argc, argv, &dev_send_buff, &dev_recv_buff, &host_buff, &host_tmp);
  int nbytes = nelem * sizeof(float);
  LOG_INFO("benchmarking send recv nbytes %d", nbytes);

  int next_peer = (context.rank + 1) % context.nranks;
  int pre_peer = (context.rank + context.nranks - 1) % context.nranks;
  double acc_time = 0;
  for (int i = 0; i < REPEAT_EXP; ++i) {
    bool c = dataIntegrityCheck(dev_recv_buff, host_buff, host_tmp, nelem);
    LOG_INFO("received buffer equals to send buffer %s (suppose false)", c? "true":"false");

    double start = timeMs();
    handle_t send_handle = ourSend(dev_send_buff, nbytes, next_peer, comm);
    handle_t recv_handle = ourRecv(dev_recv_buff, nbytes, pre_peer, comm);

    waitTask(send_handle, comm);
    waitTask(recv_handle, comm);
    acc_time += (timeMs() - start);

    c = dataIntegrityCheck(dev_recv_buff, host_buff, host_tmp, nelem);
    LOG_INFO("received buffer equals to send buffer %s (suppose true)", c? "true":"false");
    cudaMemset(dev_recv_buff, 0, nbytes);
  }
  double avg_time = acc_time / REPEAT_EXP;
  LOG_INFO("average time cost %f ms, bandwidth %f Gbps", avg_time, nbytes * 8 / avg_time / 1e6);
}

bool dataIntegrityCheck(void* dev_recv_buff, void* ref_buff, void* tmp_buff, int nelem) {
  int nbytes = nelem * sizeof(float);
  cudaMemcpy(tmp_buff, dev_recv_buff, nbytes, cudaMemcpyDefault);
  int match = memcmp(tmp_buff, ref_buff, nbytes);
  return match == 0;
}