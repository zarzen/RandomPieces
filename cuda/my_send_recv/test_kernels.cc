#include "kernels.h"

// TODO: test the send kernel that move the data from dev to host
// at host, launch a thread always move the head pointer ahead, 
// to pretend the memory buffer has been consumed
// measure the bandwidth
// -> expect to see 90Gbps
void testSendKernel() {

}

int main() {
  testSendKernel();
}