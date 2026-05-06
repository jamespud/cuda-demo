#pragma once

#include "common.cuh"

class GpuTimer {
 public:
  cudaEvent_t start_;
  cudaEvent_t stop_;

  GpuTimer() {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
  }

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { CHECK_CUDA(cudaEventRecord(start_)); }

  float stop() {
    CHECK_CUDA(cudaEventRecord(stop_));
    CHECK_CUDA(cudaEventSynchronize(stop_));

    float ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));

    return ms;
  }
};