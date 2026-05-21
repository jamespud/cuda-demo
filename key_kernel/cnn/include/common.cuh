#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define MAX_NUM_BINS 256

#define CHECK_CUDA(call)                                                                        \
    do {                                                                                        \
        cudaError_t err = call;                                                                 \
        if (err != cudaSuccess) {                                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

#define LOG_INFO(msg) std::cout << msg << std::endl

#define LOG_BENCH(name, ms, gflops)                               \
    std::cout << "[BENCH] " << name << " | Time: " << ms << " ms" \
              << " | GFLOPS: " << gflops << std::endl
