#include "sendrecv.h"


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