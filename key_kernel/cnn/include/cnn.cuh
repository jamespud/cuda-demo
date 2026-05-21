
#include <cuda_runtime.h>

/**
 * @brief Computes the convolution of the input image with a filter on the CPU.
 *
 * @param X Pointer to the input image data. (N x C x H_in x W_in)
 * @param Y Pointer to the output data. (N x M x H_out x W_out)
 * @param W Pointer to the filter weights. (M x C x K_H x K_W)
 */
void cpu_cnn(float* X, float* Y, float* W, int N, int M, int C, int H_in, int W_in, int H_out,
             int W_out, int K_H, int K_W);

void launch_native(float* d_input, float* d_out, float* d_filters, int N, int M, int C, int H_in,
                   int W_in, int H_out, int W_out, int K_H, int K_W);

void launch_unroll(float* d_input, float* d_out, float* d_filters, int N, int M, int C, int H_in,
                      int W_in, int H_out, int W_out, int K_H, int K_W);
void launch_cudnn_conv(float* d_input, float* d_out, float* d_filters, int N, int M, int C, 
                       int H_in, int W_in, int H_out, int W_out, int K_H, int K_W);