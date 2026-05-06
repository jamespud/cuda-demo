#include "conv.cuh"

__global__ void conv_naive_kernel(const float* input, const float* filter, float* output,
                                  int width, int height, int filter_size) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int radius = filter_size / 2;

    if (row < 0 || row >= height || col < 0 || col >= width) {
        return;
    }

    float sum = 0.0f;

    for (int fy = 0; fy < filter_size; fy++) {
        for (int fx = 0; fx < filter_size; fx++) {
            const int inRow = row + fy - radius;
            const int inCol = col + fx - radius;

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                sum += input[inRow * width + inCol] * filter[fy * filter_size + fx];
            }
        }
    }

    output[row * width + col] = sum;
}

void launch_conv_naive(const float* d_input, const float* filter, float* d_output, int width,
                       int height, int filter_size) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    conv_naive_kernel<<<grid, block>>>(d_input, filter, d_output, width, height, filter_size);

    CHECK_CUDA(cudaGetLastError());
}