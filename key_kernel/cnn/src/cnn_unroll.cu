#include "../include/cnn.cuh"
#include "../include/common.cuh"

#define TILE_WIDTH 16

__global__ void unroll_kernel(float* d_input, float* d_input_unroll, int N, int M, int C, int H_in,
                              int W_in, int H_out, int W_out, int K_H, int K_W) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int W_unroll = H_out * W_out;

    if (t < C * W_unroll) {
        int c = t / W_unroll;
        int w_unroll_idx = t % W_unroll;
        int h_out = w_unroll_idx / W_out;
        int w_out = w_unroll_idx % W_out;
        int w_base = c * K_H * K_W;
        for (int p = 0; p < K_H; p++) {
            for (int q = 0; q < K_W; q++) {
                int h_unroll = w_base + p * K_W + q;
                d_input_unroll[h_unroll * W_unroll + w_unroll_idx] =
                    d_input[c * H_in * W_in + (h_out + p) * W_in + (w_out + q)];
            }
        }
    }
}

// GEMM: C_matrix = A_matrix * B_matrix
__global__ void gemm_kernel(float* B_matrix, float* C_matrix, float* A_matrix, int A_rows,
                            int A_cols, int B_cols) {
    int B_rows = A_cols;

    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int ph = 0; ph < (A_cols + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < A_rows && ph * TILE_WIDTH + tx < A_cols)
            ds_A[ty][tx] = A_matrix[Row * A_cols + ph * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (ph * TILE_WIDTH + ty < B_rows && Col < B_cols)
            ds_B[ty][tx] = B_matrix[(ph * TILE_WIDTH + ty) * B_cols + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }

        __syncthreads();
    }

    // 将结果写回全局内存
    if (Row < A_rows && Col < B_cols) {
        C_matrix[Row * B_cols + Col] = Cvalue;
    }
}

void launch_unroll(float* d_input, float* d_out, float* d_filters, int N, int M, int C, int H_in,
                   int W_in, int H_out, int W_out, int K_H, int K_W) {
    const int w_flattened_h = M;              // 权重矩阵的行 (输出通道数)
    const int w_flattened_w = C * K_H * K_W;  // 权重矩阵的列 (每个卷积核的元素个数)
    const int x_flattened_h = C * K_H * K_W;  // 输入矩阵的行
    const int x_flattened_w = H_out * W_out;  // 输入矩阵的列 (输出特征图的像素总数)
    const int y_flattened_h = M;              // 输出矩阵的行
    const int y_flattened_w = H_out * W_out;  // 输出矩阵的列

    float* d_input_unroll;
    size_t unroll_size = x_flattened_h * x_flattened_w * sizeof(float);
    cudaMalloc((void**)&d_input_unroll, unroll_size);

    int num_threads = C * x_flattened_w;
    int threads_per_block = 256;
    int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    unroll_kernel<<<blocks, threads_per_block>>>(d_input, d_input_unroll, N, M, C, H_in, W_in,
                                                 H_out, W_out, K_H, K_W);
    cudaDeviceSynchronize();

    dim3 gemm_threads(TILE_WIDTH, TILE_WIDTH);
    dim3 gemm_blocks((y_flattened_w + TILE_WIDTH - 1) / TILE_WIDTH,
                     (y_flattened_h + TILE_WIDTH - 1) / TILE_WIDTH);

    gemm_kernel<<<gemm_blocks, gemm_threads>>>(d_input_unroll, d_out, d_filters, w_flattened_h,
                                               w_flattened_w, x_flattened_w);
    cudaDeviceSynchronize();

    cudaFree(d_input_unroll);
}
