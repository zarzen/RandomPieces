#include "sendrecv.h"
#include "utils.h"

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

ConnectionType_t NetConnection::getType() {
    return Net;
}

void NetConnection::sendCtrl(void* buff, size_t count) {}
void NetConnection::sendData(void* buff, size_t count) {}


Communicator::Communicator(CommunicatorArgs& context) {}

Connection& Communicator::getConnection(int peer) {
    
}

handle_t ourSend(void* dev_ptr,
                 size_t bytes_count,
                 int peer,
                 Communicator& comm) {
  // get the connection from comm

  // based on connection type launch different data movement kernel
}

handle_t ourRecv(void* dev_ptr,
                 size_t bytes_count,
                 int peer,
                 Communicator& comm) {}

void waitTask(handle_t& handle, Communicator& comm) {}