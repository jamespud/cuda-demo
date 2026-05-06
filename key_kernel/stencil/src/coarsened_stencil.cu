#include "common.cuh"
#include "stencil.cuh"

// Coarsened kernel - combines shared memory with loop coarsening
// Each thread computes multiple output elements to improve instruction-level parallelism
__global__ void coarsened_stencil_kernel(const float* __restrict__ in, float* __restrict__ out,
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

    __shared__ float s_data[COARSENING_FACTOR + 2][TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * TILE_SIZE + tx;
    int j = blockIdx.y * TILE_SIZE + ty;
    int k_base = blockIdx.z * COARSENING_FACTOR;

    int s_i = tx + 1;
    int s_j = ty + 1;

    // TODO: Load data into shared memory
    // Load multiple z-planes into shared memory
    // Handle halo regions and boundary conditions
    for (size_t z_tile = 0; z_tile < COARSENING_FACTOR + 2; z_tile++) {
        int k = k_base + z_tile - 1;
        if (k >= 0 && k < nz && i < nx && j < ny) {
            int idx = i + j * nx + k * nx * ny;
            s_data[z_tile][s_j][s_i] = in[idx];

            if (tx == 0 && i > 0) {
                s_data[z_tile][s_j][0] = in[idx - 1];
            }
            if (tx == blockDim.x - 1 && i < nx - 1) {
                s_data[z_tile][s_j][s_i + 1] = in[idx + 1];
            }
            if (ty == 0 && j > 0) {
                s_data[z_tile][0][s_i] = in[idx - nx];
            }
            if (ty == blockDim.y - 1 && j < ny - 1) {
                s_data[z_tile][s_j + 1][s_i] = in[idx + nx];
            }
        }
    }

    __syncthreads();

    // TODO: Compute stencil with loop coarsening
    // Each thread computes COARSENING_FACTOR output elements
    // Iterate over z-planes and compute stencil for each

    if (tx > 0 && tx <= TILE_SIZE && ty > 0 && ty <= TILE_SIZE) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            // TODO: Loop over z-planes with coarsening
            for (int c = 0; c < COARSENING_FACTOR; c++) {
                int k = k_base + c;

                if (k > 0 && k < nz - 1) {
                    // Access s_data[c+1][ty][tx] and neighbors
                    int idx = i + j * nx + k * nx * ny;
                    out[idx] =
                        kCenter * s_data[c + 1][s_j][s_i] +
                        kNeighbor * (s_data[c + 1][s_j][s_i - 1] + s_data[c + 1][s_j][s_i + 1] +
                                     s_data[c + 1][s_j - 1][s_i] + s_data[c + 1][s_j + 1][s_i] +
                                     s_data[c][s_j][s_i] + s_data[c + 2][s_j][s_i]);
                }
            }
        }
    }
}

// Launch function for coarsened kernel
void launch_coarsened(const float* d_in, float* d_out, int nx, int ny, int nz) {
    // TODO: Configure and launch coarsened kernel
    // Adjust grid dimensions to account for coarsening factor

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((nx + TILE_SIZE - 1) / TILE_SIZE, (ny + TILE_SIZE - 1) / TILE_SIZE,
              (nz + COARSENING_FACTOR - 1) / COARSENING_FACTOR);

    coarsened_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaGetLastError());
}