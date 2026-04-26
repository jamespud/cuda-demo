#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(err)                                               \
  if (err != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA错误 位置:%s:%d | 原因:%s\n", __FILE__, __LINE__,    \
            cudaGetErrorString(err));                                       \
    exit(EXIT_FAILURE);                                                     \
  }

__global__ void matrixVectorMultiplyKernel(const float* B, const float* C, float* A,
                               int Width) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < Width) {
    float sum = 0.0f;
    for (int col = 0; col < Width; col++) {
      sum += B[row * Width + col] * C[col];
    }
    A[row] = sum;
  }
}

void initialDeviceMatrix(float* B_h, float* C_h, float* A_h, int Width) {
  size_t matrixSize = Width * Width * sizeof(float);
  size_t vectorSize = Width * sizeof(float);

  float *B_d = NULL, *C_d = NULL, *A_d = NULL;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&B_d, matrixSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&C_d, vectorSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&A_d, vectorSize));

  CHECK_CUDA_ERROR(
      cudaMemcpy(B_d, B_h, matrixSize, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(C_d, C_h, vectorSize, cudaMemcpyHostToDevice));

  int blockSize = 256;
  int numBlocks = (Width + blockSize - 1) / blockSize;

  matrixVectorMultiplyKernel<<<numBlocks, blockSize>>>(B_d, C_d, A_d, Width);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaMemcpy(A_h, A_d, vectorSize, cudaMemcpyDeviceToHost));

  cudaFree(B_d);
  cudaFree(C_d);
  cudaFree(A_d);
}



void initializeData(float* mat, float* row, int width) {
  for (int i = 0; i < width * width; i++) {
    mat[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < width; i++) {
    row[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

// CPU矩阵向量乘法（用于验证结果）
void matrixVectorMultiplyCPU(const float* B, const float* C, float* A, int Width) {
  for (int i = 0; i < Width; i++) {
    float sum = 0.0f;
    for (int j = 0; j < Width; j++) {
      sum += B[i * Width + j] * C[j];
    }
    A[i] = sum;
  }
}

// 验证GPU结果与CPU结果是否一致
bool verifyResult(const float* A_gpu, const float* A_cpu, int Width, float eps = 1e-4f) {
  for (int i = 0; i < Width; i++) {
    if (fabs(A_gpu[i] - A_cpu[i]) > eps) {
      fprintf(stderr, "验证失败！索引%d: GPU=%.6f, CPU=%.6f\n", i, A_gpu[i], A_cpu[i]);
      return false;
    }
  }
  printf("✅ 结果验证成功！GPU与CPU计算结果一致。\n");
  return true;
}

int main() {
  const int Width = 1024; // 矩阵维度（1024x1024）
  size_t matrix_size = Width * Width * sizeof(float);
  size_t vector_size = Width * sizeof(float);


  float* B_h = (float*)malloc(matrix_size);
  float* C_h = (float*)malloc(vector_size);
  float* A_h = (float*)malloc(vector_size);

  initializeData(B_h, C_h, Width);
  initialDeviceMatrix(B_h, C_h, A_h, Width);

  // 验证结果
  float* A_cpu = (float*)malloc(vector_size);
  matrixVectorMultiplyCPU(B_h, C_h, A_cpu, Width);
  verifyResult(A_h, A_cpu, Width);
  free(A_cpu);

  free(B_h);
  free(C_h);
  free(A_h);
}
