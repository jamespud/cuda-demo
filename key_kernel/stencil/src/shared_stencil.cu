#include "../include/common.cuh"
#include "../include/stencil.cuh"

// Shared memory kernel - uses shared memory to reduce global memory accesses
__global__ void shared_stencil_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int nx, int ny, int nz) {
    // TODO: Implement shared memory 3D stencil kernel
    //
    // Key optimization: Load a tile of data into shared memory with halo
    // Each thread loads one element, plus handles halo loading
    //
    // Steps:
    // 1. Declare shared memory array (with halo: TILE_SIZE + 2)
    // 2. Calculate global and local thread indices
    // 3. Load data into shared memory (including halo)
    // 4. Synchronize threads
    // 5. Compute stencil using shared memory
    // 6. Write result to global memory

    constexpr int TILE_SIZE = 16;
    __shared__ float s_data[TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * TILE_SIZE + tx;
    int j = blockIdx.y * TILE_SIZE + ty;

    // TODO: Load data into shared memory with halo
    // - Interior points: load directly
    // - Halo points: handle boundary conditions
    // - Use conditional checks for boundaries

    __syncthreads();

    // TODO: Compute stencil using shared memory
    // Only interior threads (tx in [1, TILE_SIZE], ty in [1, TILE_SIZE])
    // should compute and write results

    if (tx > 0 && tx <= TILE_SIZE && ty > 0 && ty <= TILE_SIZE) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            // TODO: Compute 2D slice stencil for each k
            // This kernel processes one z-plane at a time
            // You'll need to loop over k or launch multiple kernel calls
        }
    }
}

// Launch function for shared memory kernel
void launch_shared(const float* d_in, float* d_out, int nx, int ny, int nz) {
    // TODO: Configure and launch shared memory kernel
    // Process one z-plane at a time or use 3D shared memory

    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((nx + TILE_SIZE - 1) / TILE_SIZE,
              (ny + TILE_SIZE - 1) / TILE_SIZE,
              1);

    // TODO: Loop over z-dimension and launch kernel for each plane
    for (int k = 1; k < nz - 1; k++) {
        // TODO: Calculate offset for current z-plane
        // Launch kernel with appropriate pointer offsets
        shared_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
        CHECK_CUDA(cudaGetLastError());
    }
}