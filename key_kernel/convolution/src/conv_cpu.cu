#include "conv.cuh"

void conv_cpu(const float* input, const float* filter, float* output, int width,
              int height, int filter_size) {
  const int radius = filter_size / 2;

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      float sum = 0.0f;

      for (int fy = 0; fy < filter_size; fy++) {
        for (int fx = 0; fx < filter_size; fx++) {
          const int inRow = row + fy - radius;
          const int inCol = col + fx - radius;

          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            sum += input[inRow * width + inCol] *
                   filter[fy * filter_size + fx];
          }
        }
      }

      output[row * width + col] = sum;
    }
  }
}