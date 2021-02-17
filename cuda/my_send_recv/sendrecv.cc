#include "sendrecv.h"



handle_t ourSend(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm) {
    // get the connection from comm

    // based on connection type launch different data movement kernel
}

handle_t ourRecv(void* dev_ptr, size_t bytes_count, int peer, Communicator& comm) {

}