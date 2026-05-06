#pragma once

#include <cuda_runtime.h>

// Stencil configuration
struct Config {
    int nx;
    int ny;
    int nz;
    int block_size;
    int coarsening_factor;

    Config(int nx_, int ny_, int nz_, int block_size_ = 16, int coarsening_factor_ = 4)
        : nx(nx_), ny(ny_), nz(nz_), block_size(block_size_), coarsening_factor(coarsening_factor_) {}
};

// CPU stencil computation
void cpu_stencil(const float* in, float* out, int nx, int ny, int nz);

// Naive CUDA kernel
void launch_naive(const float* d_in, float* d_out, int nx, int ny, int nz);

// Shared memory CUDA kernel
void launch_shared(const float* d_in, float* d_out, int nx, int ny, int nz);

// Coarsened CUDA kernel (shared memory + loop coarsening)
void launch_coarsened(const float* d_in, float* d_out, int nx, int ny, int nz);