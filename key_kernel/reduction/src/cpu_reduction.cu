#include "../include/reduction.cuh"
#include "../include/common.cuh"

void cpu_reduction(unsigned int* input, unsigned long long* out, size_t count) {
    unsigned long long sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += input[i];
    }
    *out = sum;
}