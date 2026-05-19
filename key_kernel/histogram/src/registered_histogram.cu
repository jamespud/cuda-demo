#include "../include/common.cuh"
#include "../include/histogram.cuh"

#define TILE_SIZE 32
#define COARSEN_FACTOR 8

__global__ void registered_histogram_kernel(const unsigned char* d_input, unsigned int* d_out,
                                            int channels, int width, int bins) {
    extern __shared__ unsigned int s_hist[];
    const int shared_bins = bins < 256 ? bins : 256;

    int channel = blockIdx.y;
    int tid = threadIdx.x;

    if (channel >= channels) {
        return;
    }

    // Initialize shared histogram.
    for (int bin = tid; bin < shared_bins; bin += blockDim.x) {
        s_hist[bin] = 0;
    }

    __syncthreads();

    // Each thread processes COARSEN_FACTOR consecutive elements per grid-stride step.
    int chunk_start = blockIdx.x * blockDim.x * COARSEN_FACTOR + tid * COARSEN_FACTOR;
    int chunk_stride = blockDim.x * gridDim.x * COARSEN_FACTOR;

    unsigned int accumulator = 0;
    int prev_bin = -1;

    for (int chunk = chunk_start; chunk < width; chunk += chunk_stride) {
        for (int offset = 0; offset < COARSEN_FACTOR; ++offset) {
            int idx = chunk + offset;
            if (idx < width) {
                unsigned int bin = d_input[channel * width + idx];

                if (bin == static_cast<unsigned int>(prev_bin)) {
                    ++accumulator;
                } else {
                    if (accumulator > 0) {
                        atomicAdd(&s_hist[prev_bin], accumulator);
                    }
                    prev_bin = static_cast<int>(bin);
                    accumulator = 1;
                }
            }
        }
    }

    if (accumulator > 0) {
        atomicAdd(&s_hist[prev_bin], accumulator);
    }

    __syncthreads();

    // Flush the shared histogram to global memory.
    for (int bin = tid; bin < shared_bins; bin += blockDim.x) {
        unsigned int count = s_hist[bin];
        if (count > 0) {
            atomicAdd(&d_out[channel * bins + bin], count);
        }
    }
}

void launch_registered(const unsigned char* d_input, unsigned int* d_out, int channels, int width,
                       int bins) {
    dim3 blockSize(TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE * COARSEN_FACTOR - 1) / (TILE_SIZE * COARSEN_FACTOR),
                  channels);
    size_t shared_mem = static_cast<size_t>(bins < 256 ? bins : 256) * sizeof(unsigned int);
    registered_histogram_kernel<<<gridSize, blockSize, shared_mem>>>(d_input, d_out, channels, width,
                                                                    bins);
}
