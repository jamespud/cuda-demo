#include "../include/common.cuh"
#include "../include/stencil.cuh"

// Coarsened kernel - combines shared memory with loop coarsening
// Each thread computes multiple output elements to improve instruction-level parallelism
__global__ void coarsened_stencil_kernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int nx, int ny, int nz) {
    // TODO: Implement coarsened 3D stencil kernel
    //
    // Key optimizations:
    // 1. Shared memory with halo (like shared_stencil_kernel)
    // 2. Loop coarsening: each thread computes multiple z-planes
    // 3. Better memory access patterns
    //
    // Steps:
    // 1. Declare shared memory array (with halo)
    // 2. Calculate global and local thread indices
    // 3. Load data into shared memory
    // 4. Synchronize threads
    // 5. Loop over multiple z-planes (coarsening factor)
    // 6. For each plane, compute stencil using shared memory
    // 7. Write results to global memory

    constexpr int TILE_SIZE = 16;
    constexpr int COARSENING_FACTOR = 4;  // Each thread computes 4 z-planes

    __shared__ float s_data[COARSENING_FACTOR + 2][TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * TILE_SIZE + tx;
    int j = blockIdx.y * TILE_SIZE + ty;

    // TODO: Load data into shared memory
    // Load multiple z-planes into shared memory
    // Handle halo regions and boundary conditions

    __syncthreads();

    // TODO: Compute stencil with loop coarsening
    // Each thread computes COARSENING_FACTOR output elements
    // Iterate over z-planes and compute stencil for each

    if (tx > 0 && tx <= TILE_SIZE && ty > 0 && ty <= TILE_SIZE) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            // TODO: Loop over z-planes with coarsening
            for (int c = 0; c < COARSENING_FACTOR; c++) {
                int k = blockIdx.z * COARSENING_FACTOR + c;

                if (k > 0 && k < nz - 1) {
                    // TODO: Compute stencil using shared memory
                    // Access s_data[c+1][ty][tx] and neighbors
                    int idx = i + j * nx + k * nx * ny;
                    out[idx] = in[idx];  // Placeholder
                }
            }
        }
    }
}

// Launch function for coarsened kernel
void launch_coarsened(const float* d_in, float* d_out, int nx, int ny, int nz) {
    // TODO: Configure and launch coarsened kernel
    // Adjust grid dimensions to account for coarsening factor

    constexpr int TILE_SIZE = 16;
    constexpr int COARSENING_FACTOR = 4;

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((nx + TILE_SIZE - 1) / TILE_SIZE,
              (ny + TILE_SIZE - 1) / TILE_SIZE,
              (nz + COARSENING_FACTOR - 1) / COARSENING_FACTOR);

    coarsened_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaGetLastError());
}