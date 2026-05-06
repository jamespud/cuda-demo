#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                                        \
    do {                                                                                        \
        cudaError_t err = call;                                                                 \
        if (err != cudaSuccess) {                                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

constexpr float kCenter = 0.5f;
constexpr float kNeighbor = 0.0833f;

inline void init_data(std::vector<float>& data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    for (auto& v : data) {
        v = dist(gen);
    }
}

inline bool validate_result(const std::vector<float>& ref, const std::vector<float>& gpu,
                            float eps = 1e-3f) {
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::fabs(ref[i] - gpu[i]) > eps) {
            std::cout << "Mismatch at " << i << " ref=" << ref[i] << " gpu=" << gpu[i] << std::endl;

            return false;
        }
    }

    return true;
}

