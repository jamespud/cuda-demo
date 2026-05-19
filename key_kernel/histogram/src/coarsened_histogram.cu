#include "../include/common.cuh"
#include "../include/histogram.cuh"

#define TILE_SIZE 32
#define COARSEN_FACTOR 8

__global__ void coarsened_histogram_kernel(const unsigned char* d_input, unsigned int* d_out,
                                           int channels, int width, int bins) {
    extern __shared__ unsigned int s_hist[];
    const int shared_bins = bins < 256 ? bins : 256;

    int channel = blockIdx.y;
    int thread_id = threadIdx.x;

    if (channel >= channels) {
        return;
    }

    int tile_start = blockIdx.x * blockDim.x * COARSEN_FACTOR + thread_id;
    int grid_stride = blockDim.x * gridDim.x * COARSEN_FACTOR;

    // Initialize shared histogram
    for (int bin = thread_id; bin < shared_bins; bin += blockDim.x) {
        s_hist[bin] = 0;
    }

    __syncthreads();

    for (int base = tile_start; base < width; base += grid_stride) {
        for (int offset = 0; offset < COARSEN_FACTOR; ++offset) {
            int i = base + offset * blockDim.x;
            if (i < width) {
                unsigned char bin = d_input[channel * width + i];
                atomicAdd(&s_hist[bin], 1);
            }
        }
    }

    __syncthreads();

    for (int bin = thread_id; bin < shared_bins; bin += blockDim.x) {
        unsigned int bin_value = s_hist[bin];
        if (bin_value > 0) {
            atomicAdd(&d_out[channel * bins + bin], bin_value);
        }
    }
}

void launch_coarsened(const unsigned char* d_input, unsigned int* d_out, int channels, int width,
                      int bins) {
    dim3 blockSize(TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE * COARSEN_FACTOR - 1) / (TILE_SIZE * COARSEN_FACTOR),
                  channels);
    size_t shared_mem = static_cast<size_t>(bins < 256 ? bins : 256) * sizeof(unsigned int);
    coarsened_histogram_kernel<<<gridSize, blockSize, shared_mem>>>(d_input, d_out, channels, width,
                                                                    bins);
}
