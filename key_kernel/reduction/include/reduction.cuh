#include <cuda_runtime.h>

void cpu_reduction(unsigned int* input, unsigned long long* out, size_t count);

void launch_native(unsigned int* d_input, unsigned long long* d_out, size_t count);

void launch_shared(unsigned int* d_input, unsigned long long* d_out, size_t count);

void launch_coarsened(unsigned int* d_input, unsigned long long* d_out, size_t count);
