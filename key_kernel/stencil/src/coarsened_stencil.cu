#include "common.cuh"
#include "stencil.cuh"

#define COARSENING_TILE_SIZE 32
#define COARSENING_BLOCK_SIZE (COARSENING_TILE_SIZE - 2)

/**
 * Helper function to load data into shared memory with halo regions.
 * shared: pointer to shared memory array (with halo)
 * in: pointer to global input array
 * idx: z-axis offset
 * nx, ny, nz: dimensions of the input array
 */
__device__ float load_point(const float* __restrict__ in, int global_i, int global_j, int global_k,
                            int nx, int ny, int nz) {
    if (global_i >= 0 && global_i < ny && global_j >= 0 && global_j < nx && global_k >= 0 &&
        global_k < nz) {
        return in[global_j + global_i * nx + global_k * nx * ny];
    }

    return 0.0f;
}

// Coarsened kernel - combines shared memory with loop coarsening
// Each thread computes multiple output elements to improve instruction-level parallelism
__global__ void coarsened_stencil_kernel(const float* __restrict__ in, float* __restrict__ out,
                                         int nx, int ny, int nz) {
    __shared__ float inCurr_s[COARSENING_TILE_SIZE * COARSENING_TILE_SIZE];

    int i_base = blockIdx.y * COARSENING_BLOCK_SIZE;
    int j_base = blockIdx.x * COARSENING_BLOCK_SIZE;
    int z_base = blockIdx.z * COARSENING_FACTOR;

    int idx = threadIdx.y;
    int jdx = threadIdx.x;
    int global_i = i_base + idx - 1;
    int global_j = j_base + jdx - 1;
    int shared_idx = idx * COARSENING_TILE_SIZE + jdx;
    bool compute_thread =
        idx > 0 && idx < COARSENING_TILE_SIZE - 1 && jdx > 0 && jdx < COARSENING_TILE_SIZE - 1;

    float in_prev = load_point(in, global_i, global_j, z_base - 1, nx, ny, nz);
    float in_curr = load_point(in, global_i, global_j, z_base, nx, ny, nz);
    inCurr_s[shared_idx] = in_curr;

    for (int i = 0; i < COARSENING_FACTOR; i++) {
        float in_next = load_point(in, global_i, global_j, z_base + i + 1, nx, ny, nz);
        __syncthreads();

        int gk = i + z_base;

        if (compute_thread && gk > 0 && gk < nz - 1 && global_i > 0 && global_i < ny - 1 &&
            global_j > 0 && global_j < nx - 1) {
            float neighbor_value = inCurr_s[shared_idx - 1] + inCurr_s[shared_idx + 1] +
                                   inCurr_s[shared_idx - COARSENING_TILE_SIZE] +
                                   inCurr_s[shared_idx + COARSENING_TILE_SIZE] + in_prev + in_next;

            out[global_j + global_i * nx + gk * nx * ny] =
                kCenter * in_curr + kNeighbor * neighbor_value;
        }

        __syncthreads();

        in_prev = in_curr;
        in_curr = in_next;
        inCurr_s[shared_idx] = in_curr;
    }
}

// Launch function for coarsened kernel
void launch_coarsened(const float* d_in, float* d_out, int nx, int ny, int nz) {
    dim3 block(COARSENING_TILE_SIZE, COARSENING_TILE_SIZE, 1);
    dim3 grid((nx + COARSENING_BLOCK_SIZE - 1) / COARSENING_BLOCK_SIZE,
              (ny + COARSENING_BLOCK_SIZE - 1) / COARSENING_BLOCK_SIZE,
              (nz + COARSENING_FACTOR - 1) / COARSENING_FACTOR);

    coarsened_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaGetLastError());
}