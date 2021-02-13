#include "kernels.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

void init_gpu_mem(float* cpu_mem, float* gpu_mem, int nelem) {
    // assign random vals to cpu mem
    for (int i = 0; i < nelem; i++) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        cpu_mem[i] = r;
    }
    cudaMemcpy(gpu_mem, cpu_mem, sizeof(float)*nelem, cudaMemcpyDefault);
}

int main() {
    int nelem = 1024 * 1024/4;
    void* host_mem1 = malloc(sizeof(float) * nelem);
    void* shm_cpu_gpu;
    cudaHostAlloc(&shm_cpu_gpu, nelem * sizeof(float), cudaHostAllocMapped);
    memset(shm_cpu_gpu, 0, nelem*sizeof(float));
    void* gpu_mem;
    cudaMalloc(&gpu_mem, nelem * sizeof(float));

    init_gpu_mem((float*)host_mem1, (float*)gpu_mem, nelem);
    printf("pack128 is %d floats\n", sizeof(Pack128) / sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int j = 0 ; j < 20; ++j) {
        bool pass_check = true;
        for (int i = 0; i < nelem; ++i) {
            if (((float*)host_mem1)[i] != ((float*)shm_cpu_gpu)[i])
                pass_check = false;
        }
        printf("at first equal %s \n", pass_check?"true":"false");
        cudaEventRecord(start);
        // void* args[3] = {&gpu_mem, &shm_cpu_gpu, &nelem};
        // cudaLaunchKernel((void*)dataMove, dim3(1), dim3(512), args, 0, NULL);
        int pack128_elems = nelem / (sizeof(Pack128) / sizeof(float));
        void* args[3] = {&gpu_mem, &shm_cpu_gpu, &pack128_elems};
        cudaLaunchKernel((void*)pack128Move, dim3(2), dim3(256), args, 0, NULL);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cost_ms;
        cudaEventElapsedTime(&cost_ms, start, stop);
        printf("move %d elements costs %f ms \n", nelem, cost_ms);
        // bool pass_check = true;
        pass_check = true;
        for (int i = 0; i < nelem; ++i) {
            if (((float*)host_mem1)[i] != ((float*)shm_cpu_gpu)[i])
                pass_check = false;
        }
        printf("check %s\n", pass_check?"true":"false");

        memset(shm_cpu_gpu, 0, sizeof(float)*nelem);
    }
    
    cudaDeviceSynchronize();
}