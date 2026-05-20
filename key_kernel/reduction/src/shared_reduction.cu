#include "../include/common.cuh"
#include "../include/reduction.cuh"

#define BLOCK_DIM 256

__global__ void shared_reduction_kernel(unsigned int* input, unsigned long long* out,
                                        size_t count) {
    __shared__ unsigned int input_s[BLOCK_DIM];
    
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = segment + tid;

    unsigned int val = 0;
    if (i < count) {
        val = input[i];
    }
    if (i + BLOCK_DIM < count) {
        val += input[i + BLOCK_DIM];
    }
    input_s[tid] = val;

    for (size_t stride = BLOCK_DIM / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            input_s[tid] += input_s[tid + stride];
        }
    }

    if (tid == 0) {
        atomicAdd(out, input_s[0]);
    }
}

void launch_shared(unsigned int* d_input, unsigned long long* d_out, size_t count) {
    int num_blocks = (count + 2 * BLOCK_DIM - 1) / (2 * BLOCK_DIM);
    shared_reduction_kernel<<<num_blocks, BLOCK_DIM>>>(d_input, d_out, count);
    CHECK_CUDA(cudaGetLastError());
}