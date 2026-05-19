#include "../include/common.cuh"
#include "../include/histogram.cuh"

#define TILE_SIZE 32

__global__ void shared_histogram_kernel(const unsigned char* d_input, unsigned int* d_out,
                                        int channels, int width, int bins) {
    extern __shared__ unsigned int s_hist[];
    const int shared_bins = bins < 256 ? bins : 256;

    int i_base = blockIdx.x * TILE_SIZE;
    int i_local = threadIdx.x;
    int i = i_base + i_local;

    // Initialize shared histogram
    for (int j = i_local; j < shared_bins; j += TILE_SIZE) {
        s_hist[j] = 0;
    }

    __syncthreads();

    if (i < width) {
        int bin = d_input[blockIdx.y * width + i];
        atomicAdd(&s_hist[bin], 1);
    }

    __syncthreads();

    for (int i = i_local; i < shared_bins; i += TILE_SIZE) {
        unsigned int binValue = s_hist[i];
        if (binValue > 0) {
            atomicAdd(&d_out[blockIdx.y * bins + i], binValue);
        }
    }
}

void launch_shared(const unsigned char* d_input, unsigned int* d_out, int channels, int width,
                   int bins) {
    dim3 blockSize(TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, channels);
    size_t shared_mem = static_cast<size_t>(bins < 256 ? bins : 256) * sizeof(unsigned int);
    shared_histogram_kernel<<<gridSize, blockSize, shared_mem>>>(d_input, d_out, channels, width,
                                                                 bins);
}
