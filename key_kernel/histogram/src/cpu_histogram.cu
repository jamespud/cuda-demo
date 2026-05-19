#include "../include/common.cuh"
#include "../include/histogram.cuh"

void cpu_histogram(const unsigned char* input, unsigned int* out, int channels, int width,
                   int bins) {
    for (size_t i = 0; i < channels; i++) {
        for (size_t j = 0; j < width; j++) {
            unsigned char bin = input[i * width + j];
            out[i * bins + bin]++;
        }
    }
}