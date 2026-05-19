#include "../include/common.cuh"
#include "../include/histogram.cuh"

__global__ void native_histogram_kernel(const unsigned char* d_input, unsigned int* d_out,
                                        int channels, int width, int bins) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < channels && j < width) {
        unsigned int bin = d_input[i * width + j];
        atomicAdd(&d_out[i * bins + bin], 1);
    }
}

void launch_native(const unsigned char* d_input, unsigned int* d_out, int channels, int width,
                   int bins) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (channels + blockSize.y - 1) / blockSize.y);
    native_histogram_kernel<<<gridSize, blockSize>>>(d_input, d_out, channels, width, bins);
}