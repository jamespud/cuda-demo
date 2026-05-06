#include "conv.cuh"

#include <cmath>

bool check_result(const float* ref, const float* gpu, int size) {
  for (int i = 0; i < size; i++) {
    float diff = std::fabs(ref[i] - gpu[i]);

    if (diff > 1e-3f) {
      std::cout << "Mismatch at " << i << " ref=" << ref[i] << " gpu=" << gpu[i]
                << std::endl;

      return false;
    }
  }

  return true;
}