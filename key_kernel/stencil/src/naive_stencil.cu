#include "../include/common.cuh"
#include "../include/stencil.cuh"

// Naive CUDA kernel - one thread per output element
__global__ void naive_stencil_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int nx, int ny, int nz) {
    // TODO: Implement naive 3D stencil kernel
    // Each thread computes one output element
    //
    // Steps:
    // 1. Calculate global thread indices (i, j, k)
    // 2. Check boundary conditions
    // 3. Compute linear index
    // 4. Apply 7-point stencil formula
    // 5. Write result to output

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // TODO: Check boundaries and compute stencil
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1) {
        int idx = i + j * nx + k * nx * ny;
        // TODO: Compute stencil here
        out[idx] = in[idx];  // Placeholder
    }
}

// Launch function for naive kernel
void launch_naive(const float* d_in, float* d_out, int nx, int ny, int nz) {
    // TODO: Configure and launch naive kernel
    // Use dim3 for block and grid dimensions
    // Typical block size: 16x16x1 or 8x8x8

    dim3 block(16, 16, 1);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    naive_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaGetLastError());
}