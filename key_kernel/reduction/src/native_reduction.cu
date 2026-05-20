#include "../include/common.cuh"
#include "../include/reduction.cuh"

#define THREADS_PER_BLOCK 256

__global__ void native_reduction_kernel(unsigned int* input, unsigned long long* out,
                                        size_t count) {
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int i = 2 * (offset + tid);
    for (size_t stride = 1; stride < 2 * blockDim.x; stride *= 2) {
        if (tid % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, input[i]);
    }
}

void launch_native(unsigned int* d_input, unsigned long long* d_out, size_t count) {
    int total_threads = (count + 1) / 2;
    int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    native_reduction_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_out, count);
    CHECK_CUDA(cudaGetLastError());
}