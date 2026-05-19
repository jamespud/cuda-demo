
#include <cuda_runtime.h>


/**
 * @brief Computes the histogram of the input image on the CPU.
 * 
 * @param input Pointer to the input image data. (channels x width)
 * @param out Pointer to the output histogram data. (channels x bins)
 * @param channels Number of channels in the input image.
 * @param width Width of the input image.
 * @param bins Number of bins in the histogram.
 */
void cpu_histogram(const unsigned char* input, unsigned int* out, int channels, int width, int bins);

void launch_native(const unsigned char* d_input, unsigned int* d_out, int channels, int width, int bins);

void launch_shared(const unsigned char* d_input, unsigned int* d_out, int channels, int width, int bins);

void launch_coarsened(const unsigned char* d_input, unsigned int* d_out, int channels, int width, int bins);

void launch_registered(const unsigned char* d_input, unsigned int* d_out, int channels, int width, int bins);