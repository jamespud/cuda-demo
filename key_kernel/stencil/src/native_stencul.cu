#include "../include/common.cuh"

__global__ void stencil_naive_kernel(const float* in, float* out, int nx, int ny, int nz) {
    // TODO:
    //
    // 1. 计算 x/y/z
    // 2. boundary check
    // 3. flatten index
    // 4. 计算7-point stencil
}

void launch_naive(const float* d_in, float* d_out, int nx, int ny, int nz) {
    dim3 block(8, 8, 8);

    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    stencil_naive_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);

    CHECK_CUDA(cudaGetLastError());
}