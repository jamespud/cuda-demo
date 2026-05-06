#include "../include/common.cuh"
#include "../include/stencil.cuh"

// Shared memory kernel - uses a 2D tile per z-plane
__global__ void shared_stencil_kernel(const float* in, float* out, int nx, int ny, int k) {
    __shared__ float s_data[TILE_SIZE + 2][TILE_SIZE + 2];

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

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * TILE_SIZE + tx;
    int j = blockIdx.y * TILE_SIZE + ty;

    int s_i = tx + 1;
    int s_j = ty + 1;
    int k_offset = k * nx * ny;

    // TODO: Load data into shared memory with halo
    // - Interior points: load directly
    // - Halo points: handle boundary conditions
    // - Use conditional checks for boundaries
    if (i < nx && j < ny) {
        int idx = i + j * nx + k_offset;
        s_data[s_j][s_i] = in[idx];

        if (tx == 0 && i > 0) {
            s_data[s_j][0] = in[idx - 1];
        }
        if (tx == blockDim.x - 1 && i < nx - 1) {
            s_data[s_j][s_i + 1] = in[idx + 1];
        }
        if (ty == 0 && j > 0) {
            s_data[0][s_i] = in[idx - nx];
        }
        if (ty == blockDim.y - 1 && j < ny - 1) {
            s_data[s_j + 1][s_i] = in[idx + nx];
        }
    }

    // TODO: Compute stencil using shared memory
    // Only interior threads (tx in [1, TILE_SIZE], ty in [1, TILE_SIZE])
    // should compute and write results

    __syncthreads();

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = i + j * nx + k_offset;
        out[idx] = kCenter * s_data[s_j][s_i] +
                   kNeighbor * (s_data[s_j][s_i - 1] + s_data[s_j][s_i + 1] + s_data[s_j - 1][s_i] +
                                s_data[s_j + 1][s_i] + in[idx - nx * ny] + in[idx + nx * ny]);
    }
}

// Launch function for shared memory kernel
void launch_shared(const float* d_in, float* d_out, int nx, int ny, int nz) {
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((nx + TILE_SIZE - 1) / TILE_SIZE, (ny + TILE_SIZE - 1) / TILE_SIZE, 1);

    for (int k = 1; k < nz - 1; ++k) {
        shared_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, k);
        CHECK_CUDA(cudaGetLastError());
    }
}