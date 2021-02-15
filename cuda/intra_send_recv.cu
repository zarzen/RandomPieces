#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "sem_wrapper.hpp"
#include "shared_memory.hpp"
#include <cstdlib>
#include "kernels.h"
#include <cstring>
#include <cassert>

template<typename T>
void initBuffers(void** host_buff, void** gpu_send, void** gpu_recv, int nelem) {
    *host_buff = malloc(sizeof(T) * nelem);
    cudaMalloc(gpu_send, sizeof(T) * nelem);
    cudaMalloc(gpu_recv, sizeof(T) * nelem);

    T* rand_arr = (T*) *host_buff;
    for (int i = 0; i < nelem; ++i) {
        T e = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        rand_arr[i] = e;
    }
    cudaMemcpy(gpu_send, host_buff, sizeof(T) * nelem, cudaMemcpyDefault);
}

int main(int argc, char const *argv[]) {
    if (argc < 4) {
        printf("require input n_devices, local device idx and nelem\n");
        return -1;
    }
    int n_devices = std::stoi(argv[1]);
    int device_id = std::stoi(argv[2]);
    int nelem =  std::stoi(argv[3]);
    srand(123);

    cudaSetDevice(device_id);
    SharedMemory shm(SharedMemory::OpenType::e_open_or_create, "shm_send_recv");
    int slot_size = sizeof(int) + sizeof(cudaIpcMemHandle_t);
    shm.truncate(slot_size * n_devices);
    // SemaphoreMutex sem(NamedSemaphore::OpenType::e_open_or_create, "sem_send_recv");

    void* host_buff;
    void* gpu_send_buff;
    void* gpu_recv_buff;

    initBuffers<float>(&host_buff, &gpu_send_buff, &gpu_recv_buff, nelem);

    int n_pack = sizeof(Pack128) / sizeof(float);
    assert(((nelem / n_pack) % UNROLL) == 0);
    int pack128_nelem = nelem / n_pack;

    // prepare memory handler for gpu_recv_buff, then send the memory handler via shm
    // so the peer could get the 
    cudaIpcMemHandle_t ipc_handle;
    CUDACHECK(cudaIpcGetMemHandle(&ipc_handle, gpu_recv_buff));
    // write the memory_handle value to shm
    void* shm_ptr = (char*)shm.get_ptr() + device_id * slot_size;
    cudaIpcMemHandle_t* handle_ptr = (cudaIpcMemHandle_t*)((char*)shm_ptr + sizeof(int)); 
    *handle_ptr = ipc_handle;
    int* flag_ptr = (int*)shm_ptr;
    *flag_ptr = 1;

    // send to next peer
    int peer = (device_id + 1) % n_devices;
    void* peer_shm_ptr = (char*)shm.get_ptr() + peer * slot_size;
    int* peer_flag_ptr = (int*) peer_shm_ptr;
    while(*peer_flag_ptr < 1) {
    }
    cudaIpcMemHandle_t peer_m_handle;
    memcpy(&peer_m_handle, (char*)peer_shm_ptr + sizeof(int), sizeof(cudaIpcMemHandle_t));

    void* peer_ipc_ptr;
    CUDACHECK(cudaIpcOpenMemHandle(&peer_ipc_ptr, peer_m_handle, cudaIpcMemLazyEnablePeerAccess));
    printf("opened peer %d's recv ptr\n", peer);

    int repeat = 20;
    float acc_time = 0.0;
    cudaEvent_t start, stop;
    cudaStream_t stream;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int j = 0; j < repeat; ++j) {
        void* kernel_args[3] = {&gpu_send_buff, &peer_ipc_ptr, &pack128_nelem};
        cudaEventRecord(start);
        CUDACHECK(cudaLaunchKernel((void*)pack128Move, dim3(1), dim3(256), kernel_args, 0, stream));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cost_ms;
        cudaEventElapsedTime(&cost_ms, start, stop);
        acc_time+=cost_ms;
    }
    float avg_time = acc_time / repeat;
    printf("average send time %f ms, bandwidth %f Gbps \n", avg_time, nelem*sizeof(float)*8 / avg_time / 1e6);

    return 0;
}
