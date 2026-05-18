#pragma once

#include "common.cuh"

constexpr int IN_TILE_DIM = 32;
constexpr int MAX_FILTER_SIZE = IN_TILE_DIM;

extern __constant__ float d_filter_constant[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

void conv_cpu(const float* input, const float* filter, float* output, int width,
              int height, int filter_size);

void launch_conv_native(const float* d_input, const float* filter, float* d_output, int width,
                       int height, int filter_size);

void copy_filter_to_constant(const float* filter, int filter_size);

void launch_conv_constant(const float* d_input, float* d_output, int width, int height,
                          int filter_size);

void launch_conv_tiled(const float* d_input, float* d_output, int width, int height,
                       int filter_size);

bool check_result(const float* ref, const float* gpu, int size);