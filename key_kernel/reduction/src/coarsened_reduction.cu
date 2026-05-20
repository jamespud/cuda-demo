#include "../include/common.cuh"
#include "../include/reduction.cuh"

#define BLOCK_DIM 256
#define COARSE_FACTOR 4

__global__ void coarsened_reduction_kernel(unsigned int* input, unsigned long long* out,
                                        size_t count) {
    __shared__ unsigned int input_s[BLOCK_DIM];
    
    unsigned int segment = 2 * blockDim.x * blockIdx.x * COARSE_FACTOR;
    unsigned int tid = threadIdx.x;
    unsigned int i = segment + tid;

    unsigned int val = input[i];

    for (size_t tile = 1; tile < COARSE_FACTOR * 2; tile++) {
        if (i + tile * blockDim.x < count) {
            val += input[i + tile * blockDim.x];
        }
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

void launch_coarsened(unsigned int* d_input, unsigned long long* d_out, size_t count) {
    int num_blocks = (count + 2 * BLOCK_DIM * COARSE_FACTOR - 1) / (2 * BLOCK_DIM * COARSE_FACTOR);
    coarsened_reduction_kernel<<<num_blocks, BLOCK_DIM>>>(d_input, d_out, count);
    CHECK_CUDA(cudaGetLastError());
}