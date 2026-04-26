// ==================== 1. 头文件引入 ====================
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ==================== 2. CUDA错误检查宏（规范写法） ====================
#define CHECK_CUDA_ERROR(err)                                              \
  if (err != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA错误 位置:%s:%d | 原因:%s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err));                                      \
    exit(EXIT_FAILURE);                                                    \
  }

// ==================== 3. 核函数实现区 ====================

/**
 * @brief 原始核函数：一个线程计算输出矩阵的一个元素（作为参考基准）
 * @param M_d 设备端输入矩阵M (Width x Width)
 * @param N_d 设备端输入矩阵N (Width x Width)
 * @param P_d 设备端输出矩阵P = M * N (Width x Width)
 * @param Width 矩阵宽度
 */
__global__ void matrixMulElementKernel(const float* M_d, const float* N_d,
                                       float* P_d, int Width) {
  // 二维线程映射：计算当前线程对应的行和列
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;

  // 越界保护
  if (row < Width && col < Width) {
    float sum = 0.0f;
    // 计算内积：P[row][col] = sum(M[row][k] * N[k][col])
    for (int k = 0; k < Width; k++) {
      sum += M_d[row * Width + k] * N_d[k * Width + col];
    }
    // 写入结果
    P_d[row * Width + col] = sum;
  }
}

/**
 * @brief 行优先核函数：一个线程计算输出矩阵的一整行
 * @param M_d 设备端输入矩阵M (Width x Width)
 * @param N_d 设备端输入矩阵N (Width x Width)
 * @param P_d 设备端输出矩阵P = M * N (Width x Width)
 * @param Width 矩阵宽度
 */
__global__ void matrixMulRowKernel(const float* M_d, const float* N_d,
                                   float* P_d, int Width) {
  // 一维线程映射：计算当前线程处理的行号
  int row = blockIdx.x * 256 + threadIdx.x;

  // 越界保护：只处理有效范围内的行
  if (row < Width) {
    // 遍历当前行的所有列
    for (int col = 0; col < Width; col++) {
      float sum = 0.0f;
      // 计算M的第row行和N的第col列的内积
      for (int k = 0; k < Width; k++) {
        // 行主序访问：
        // M[row][k] = M_d[row * Width + k] (同一行连续存储)
        // N[k][col] = N_d[k * Width + col] (同一列需跨行访问)
        sum += M_d[row * Width + k] * N_d[k * Width + col];
      }
      // 写入当前行当前列的结果
      P_d[row * Width + col] = sum;
    }
  }
}

/**
 * @brief 列优先核函数：一个线程计算输出矩阵的一整列
 * @param M_d 设备端输入矩阵M (Width x Width)
 * @param N_d 设备端输入矩阵N (Width x Width)
 * @param P_d 设备端输出矩阵P = M * N (Width x Width)
 * @param Width 矩阵宽度
 */
__global__ void matrixMulColKernel(const float* M_d, const float* N_d,
                                   float* P_d, int Width) {
  // 一维线程映射：计算当前线程处理的列号
  int col = blockIdx.x * 256 + threadIdx.x;

  // 越界保护：只处理有效范围内的列
  if (col < Width) {
    // 遍历当前列的所有行
    for (int row = 0; row < Width; row++) {
      float sum = 0.0f;
      // 计算M的第row行和N的第col列的内积
      for (int k = 0; k < Width; k++) {
        // 行主序访问：
        // M[row][k] = M_d[row * Width + k]
        // N[k][col] = N_d[k * Width + col]
        sum += M_d[row * Width + k] * N_d[k * Width + col];
      }
      // 写入当前列当前行的结果
      P_d[row * Width + col] = sum;
    }
  }
}

// ==================== 4. CPU端参考计算区（用于结果校验） ====================

/**
 * @brief CPU端矩阵乘法：串行计算，生成参考结果
 * @param M_h 主机端输入矩阵M
 * @param N_h 主机端输入矩阵N
 * @param P_ref 主机端输出参考矩阵
 * @param Width 矩阵宽度
 */
void matrixMulCPU(const float* M_h, const float* N_h, float* P_ref, int Width) {
  for (int row = 0; row < Width; row++) {
    for (int col = 0; col < Width; col++) {
      float sum = 0.0f;
      for (int k = 0; k < Width; k++) {
        sum += M_h[row * Width + k] * N_h[k * Width + col];
      }
      P_ref[row * Width + col] = sum;
    }
  }
}

// ==================== 5. 主机端存根函数区（内存管理+核函数调用+计时）
// ====================

/**
 * @brief 结果校验函数：对比GPU结果和CPU参考结果
 * @param P_h GPU计算结果
 * @param P_ref CPU参考结果
 * @param Width 矩阵宽度
 * @param kernelName 核函数名称（用于打印）
 */
void verifyResult(const float* P_h, const float* P_ref, int Width,
                  const char* kernelName) {
  int errorCount = 0;
  const float eps = 1e-3f;  // 浮点误差容忍度

  for (int i = 0; i < Width * Width; i++) {
    if (fabs(P_h[i] - P_ref[i]) > eps) {
      errorCount++;
      if (errorCount < 10)  // 只打印前10个错误，避免刷屏
      {
        printf("%s 计算错误 位置:%d | GPU结果:%f | CPU结果:%f\n", kernelName, i,
               P_h[i], P_ref[i]);
      }
    }
  }

  if (errorCount == 0) {
    printf("✅ %s 测试通过！所有元素计算正确\n", kernelName);
  } else {
    printf("❌ %s 测试失败！错误元素数量:%d\n", kernelName, errorCount);
  }
}

/**
 * @brief 存根函数：调用原始单元素核函数
 */
void matrixMulElement(const float* M_h, const float* N_h, float* P_h,
                      int Width) {
  int byteSize = Width * Width * sizeof(float);
  float *M_d = NULL, *N_d = NULL, *P_d = NULL;

  // 分配设备内存
  CHECK_CUDA_ERROR(cudaMalloc((void**)&M_d, byteSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&N_d, byteSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&P_d, byteSize));

  // 数据从主机拷贝到设备
  CHECK_CUDA_ERROR(cudaMemcpy(M_d, M_h, byteSize, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(N_d, N_h, byteSize, cudaMemcpyHostToDevice));

  // 配置核函数启动参数：二维16x16块
  dim3 blockSize(16, 16);
  dim3 gridSize((Width + blockSize.x - 1) / blockSize.x,
                (Width + blockSize.y - 1) / blockSize.y);

  // 创建CUDA事件用于精确计时
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // 记录开始时间
  CHECK_CUDA_ERROR(cudaEventRecord(start));

  // 调用核函数
  matrixMulElementKernel<<<gridSize, blockSize>>>(M_d, N_d, P_d, Width);
  CHECK_CUDA_ERROR(cudaGetLastError());       // 检查核函数启动错误
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());  // 等待所有线程执行完成

  // 记录结束时间
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  // 计算并打印执行时间
  float milliseconds = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("原始核函数（一个线程一个元素）执行时间: %.2f ms\n", milliseconds);

  // 销毁事件
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  // 结果从设备拷贝回主机
  CHECK_CUDA_ERROR(cudaMemcpy(P_h, P_d, byteSize, cudaMemcpyDeviceToHost));

  // 释放设备内存
  CHECK_CUDA_ERROR(cudaFree(M_d));
  CHECK_CUDA_ERROR(cudaFree(N_d));
  CHECK_CUDA_ERROR(cudaFree(P_d));
}

/**
 * @brief 存根函数：调用行优先核函数
 */
void matrixMulRow(const float* M_h, const float* N_h, float* P_h, int Width) {
  int byteSize = Width * Width * sizeof(float);
  float *M_d = NULL, *N_d = NULL, *P_d = NULL;

  // 分配设备内存
  CHECK_CUDA_ERROR(cudaMalloc((void**)&M_d, byteSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&N_d, byteSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&P_d, byteSize));

  // 数据从主机拷贝到设备
  CHECK_CUDA_ERROR(cudaMemcpy(M_d, M_h, byteSize, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(N_d, N_h, byteSize, cudaMemcpyHostToDevice));

  // 【关键配置】行优先核函数执行参数：
  // - 一维块，每个块256个线程
  // - 网格大小 = 向上取整(Width / 256)
  dim3 blockSize(256);
  dim3 gridSize((Width + blockSize.x - 1) / blockSize.x);

  // CUDA事件计时
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));

  // 调用核函数
  matrixMulRowKernel<<<gridSize, blockSize>>>(M_d, N_d, P_d, Width);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 计时结束
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("行优先核函数（一个线程一行）执行时间: %.2f ms\n", milliseconds);

  // 销毁事件
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  // 结果拷回并释放内存
  CHECK_CUDA_ERROR(cudaMemcpy(P_h, P_d, byteSize, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(M_d));
  CHECK_CUDA_ERROR(cudaFree(N_d));
  CHECK_CUDA_ERROR(cudaFree(P_d));
}

/**
 * @brief 存根函数：调用列优先核函数
 */
void matrixMulCol(const float* M_h, const float* N_h, float* P_h, int Width) {
  int byteSize = Width * Width * sizeof(float);
  float *M_d = NULL, *N_d = NULL, *P_d = NULL;

  // 分配设备内存
  CHECK_CUDA_ERROR(cudaMalloc((void**)&M_d, byteSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&N_d, byteSize));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&P_d, byteSize));

  // 数据从主机拷贝到设备
  CHECK_CUDA_ERROR(cudaMemcpy(M_d, M_h, byteSize, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(N_d, N_h, byteSize, cudaMemcpyHostToDevice));

  // 【关键配置】列优先核函数执行参数：
  // - 一维块，每个块256个线程
  // - 网格大小 = 向上取整(Width / 256)
  dim3 blockSize(256);
  dim3 gridSize((Width + blockSize.x - 1) / blockSize.x);

  // CUDA事件计时
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));

  // 调用核函数
  matrixMulColKernel<<<gridSize, blockSize>>>(M_d, N_d, P_d, Width);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 计时结束
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("列优先核函数（一个线程一列）执行时间: %.2f ms\n", milliseconds);

  // 销毁事件
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  // 结果拷回并释放内存
  CHECK_CUDA_ERROR(cudaMemcpy(P_h, P_d, byteSize, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(M_d));
  CHECK_CUDA_ERROR(cudaFree(N_d));
  CHECK_CUDA_ERROR(cudaFree(P_d));
}

// ==================== 6. 主函数 ====================
int main() {
  // 矩阵宽度（可自由修改，建议使用512/1024，避免CPU计算时间过长）
  const int Width = 1024;
  printf("矩阵乘法对比测试 | 矩阵大小:%dx%d\n\n", Width, Width);

  // 分配主机端内存
  float* M_h = (float*)malloc(Width * Width * sizeof(float));
  float* N_h = (float*)malloc(Width * Width * sizeof(float));
  float* P_h_element = (float*)malloc(Width * Width * sizeof(float));
  float* P_h_row = (float*)malloc(Width * Width * sizeof(float));
  float* P_h_col = (float*)malloc(Width * Width * sizeof(float));
  float* P_ref = (float*)malloc(Width * Width * sizeof(float));

  // 初始化主机端数据：简单初始化，方便调试
  for (int i = 0; i < Width * Width; i++) {
    M_h[i] = (float)(i % 10);
    N_h[i] = (float)(i % 10);
  }

  // 1. 用CPU计算参考结果
  printf("正在用CPU计算参考结果...\n");
  matrixMulCPU(M_h, N_h, P_ref, Width);
  printf("CPU参考结果计算完成\n\n");

  // 2. 测试原始单元素核函数
  printf("正在测试原始核函数（一个线程一个元素）...\n");
  matrixMulElement(M_h, N_h, P_h_element, Width);
  verifyResult(P_h_element, P_ref, Width, "原始核函数（一个线程一个元素）");
  printf("\n");

  // 3. 测试行优先核函数
  printf("正在测试行优先核函数（一个线程一行）...\n");
  matrixMulRow(M_h, N_h, P_h_row, Width);
  verifyResult(P_h_row, P_ref, Width, "行优先核函数（一个线程一行）");
  printf("\n");

  // 4. 测试列优先核函数
  printf("正在测试列优先核函数（一个线程一列）...\n");
  matrixMulCol(M_h, N_h, P_h_col, Width);
  verifyResult(P_h_col, P_ref, Width, "列优先核函数（一个线程一列）");
  printf("\n");

  // 释放主机端内存
  free(M_h);
  free(N_h);
  free(P_h_element);
  free(P_h_row);
  free(P_h_col);
  free(P_ref);

  // 重置CUDA设备
  cudaDeviceReset();

  return 0;
}