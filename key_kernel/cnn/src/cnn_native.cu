#include "../include/cnn.cuh"
#include "../include/common.cuh"

#define TILE_WIDTH 16

__global__ void native_cnn(const float* d_input, float* d_out, float* d_filters, int N, int M,
                           int C, int H_in, int W_in, int H_out, int W_out, int K_H, int K_W,
                           int H_grid, int W_grid) {
    int n = blockIdx.z;
    int m = blockIdx.x;

    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    if (h >= H_out || w >= W_out) return;

    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K_H; p++) {
            for (int q = 0; q < K_W; q++) {
                acc += d_input[n * C * H_in * W_in + c * H_in * W_in + (h + p) * W_in + (w + q)] *
                       d_filters[m * C * K_H * K_W + c * K_H * K_W + p * K_W + q];
            }
        }
    }
    d_out[n * M * H_out * W_out + m * H_out * W_out + h * W_out + w] = acc;
}

void launch_native(float* d_input, float* d_out, float* d_filters, int N, int M, int C, int H_in,
                   int W_in, int H_out, int W_out, int K_H, int K_W) {
    int h_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int w_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    dim3 gridDim(M, h_grid * w_grid, N);
    native_cnn<<<gridDim, blockDim>>>(d_input, d_out, d_filters, N, M, C, H_in, W_in, H_out, W_out,
                                      K_H, K_W, h_grid, w_grid);
}