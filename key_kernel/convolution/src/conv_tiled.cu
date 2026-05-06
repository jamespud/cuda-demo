#include "conv.cuh"

__global__ void conv_tiled_kernel(const float* input, float* output, int width,
                                  int height, int filter_size) {
    const int radius = filter_size / 2;
    const int output_tile_dim = IN_TILE_DIM - filter_size + 1;

    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];

    const int outBaseRow = blockIdx.y * output_tile_dim;
    const int outBaseCol = blockIdx.x * output_tile_dim;

    const int sRow = threadIdx.y;
    const int sCol = threadIdx.x;

    const int inRow = outBaseRow + sRow - radius;
    const int inCol = outBaseCol + sCol - radius;

    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
        tile[sRow][sCol] = input[inRow * width + inCol];
    } else {
        tile[sRow][sCol] = 0.0f;
    }

    __syncthreads();

    if (sRow < output_tile_dim && sCol < output_tile_dim) {
        const int outRow = outBaseRow + sRow;
        const int outCol = outBaseCol + sCol;

        if (outRow < height && outCol < width) {
            float sum = 0.0f;
            for (int r = 0; r < filter_size; ++r) {
                for (int c = 0; c < filter_size; ++c) {
                    sum += tile[sRow + r][sCol + c] *
                           d_filter_constant[r * filter_size + c];
                }
            }
            output[outRow * width + outCol] = sum;
        }
    }
}

void launch_conv_tiled(const float* d_input, float* d_output, int width, int height,
                       int filter_size) {
    const int output_tile_dim = IN_TILE_DIM - filter_size + 1;
    dim3 block(IN_TILE_DIM, IN_TILE_DIM);
    dim3 grid((width + output_tile_dim - 1) / output_tile_dim,
              (height + output_tile_dim - 1) / output_tile_dim);

    conv_tiled_kernel<<<grid, block>>>(d_input, d_output, width, height, filter_size);

    CHECK_CUDA(cudaGetLastError());
}