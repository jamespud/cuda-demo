#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(err)                                              \
  if (err != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA错误 位置:%s:%d | 原因:%s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err));                                      \
    exit(EXIT_FAILURE);                                                    \
  }

// ==========================================
// 配置参数
// ==========================================
const int MATRIX_SIZE = 8192; // 矩阵大小：1024x1024
const int TILE_WIDTH = 8;     // 分块大小：16x16

// ==========================================
// 内核1：简单矩阵乘法（无共享内存，无同步）
// ==========================================
__global__ void simpleMatrixMultiply(const float* M, const float* N, float* P, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < Width && Col < Width) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; k++) {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }
        P[Row * Width + Col] = Pvalue;
    }
}

// ==========================================
// 内核2：分块矩阵乘法（使用共享内存 + __syncthreads()）
// ==========================================
__global__ void tiledMatrixMultiply(const float* M, const float* N, float* P, int Width) {
    // 声明共享内存
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 计算当前线程负责的输出元素坐标
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // 循环处理每个分块
    int num_phases = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < num_phases; ph++) {
        // -----------------------------------------------------------
        // 步骤1：协作加载当前分块的 M 和 N 到共享内存（带边界检查）
        // -----------------------------------------------------------
        if (Row < Width && (ph * TILE_WIDTH + tx) < Width) {
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f; // 越界填0
        }

        if ((ph * TILE_WIDTH + ty) < Width && Col < Width) {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        } else {
            Nds[ty][tx] = 0.0f; // 越界填0
        }

        // -----------------------------------------------------------
        // 步骤2：【关键同步点1】等所有线程都加载完共享内存
        // -----------------------------------------------------------
        __syncthreads();

        // -----------------------------------------------------------
        // 步骤3：用共享内存里的数据计算当前分块的贡献
        // -----------------------------------------------------------
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        // -----------------------------------------------------------
        // 步骤4：【关键同步点2】等所有线程都用完当前分块的数据
        // -----------------------------------------------------------
        __syncthreads();
    }

    // 写回最终结果
    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }
}

// ==========================================
// 辅助函数：初始化矩阵（随机浮点数）
// ==========================================
void initializeMatrix(float* mat, int size) {
    for (int i = 0; i < size * size; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX; // 0~1 随机数
    }
}

// ==========================================
// 辅助函数：验证结果（对比两个矩阵是否一致）
// ==========================================
bool verifyResult(const float* P1, const float* P2, int size, float eps = 1e-4f) {
    for (int i = 0; i < size * size; i++) {
        if (fabs(P1[i] - P2[i]) > eps) {
            fprintf(stderr, "❌ 验证失败！索引%d: 简单版=%.6f, 分块版=%.6f\n", i, P1[i], P2[i]);
            return false;
        }
    }
    printf("✅ 结果验证成功！两个版本计算结果一致。\n\n");
    return true;
}

// ==========================================
// 主函数
// ==========================================
int main() {
    size_t matrix_bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    printf("🔢 矩阵大小: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("🔲 分块大小: %dx%d\n\n", TILE_WIDTH, TILE_WIDTH);

    // 1. 分配主机内存
    float *M_h = (float*)malloc(matrix_bytes);
    float *N_h = (float*)malloc(matrix_bytes);
    float *P_simple_h = (float*)malloc(matrix_bytes);
    float *P_tiled_h = (float*)malloc(matrix_bytes);

    // 2. 初始化输入数据
    initializeMatrix(M_h, MATRIX_SIZE);
    initializeMatrix(N_h, MATRIX_SIZE);

    // 3. 分配设备内存
    float *M_d, *N_d, *P_simple_d, *P_tiled_d;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&M_d, matrix_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&N_d, matrix_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&P_simple_d, matrix_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&P_tiled_d, matrix_bytes));

    // 4. 拷贝数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(M_d, M_h, matrix_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(N_d, N_h, matrix_bytes, cudaMemcpyHostToDevice));

    // 5. 配置内核启动参数
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((MATRIX_SIZE + TILE_WIDTH - 1) / TILE_WIDTH,
                  (MATRIX_SIZE + TILE_WIDTH - 1) / TILE_WIDTH);

    // 6. 测试【简单版本】性能
    cudaEvent_t start_simple, stop_simple;
    float time_simple_ms;
    cudaEventCreate(&start_simple);
    cudaEventCreate(&stop_simple);

    printf("🚀 开始运行【简单版本】（无共享内存）...\n");
    cudaEventRecord(start_simple);
    simpleMatrixMultiply<<<gridDim, blockDim>>>(M_d, N_d, P_simple_d, MATRIX_SIZE);
    cudaEventRecord(stop_simple);
    cudaEventSynchronize(stop_simple);
    cudaEventElapsedTime(&time_simple_ms, start_simple, stop_simple);
    printf("⏱️  简单版本耗时: %.2f ms\n\n", time_simple_ms);

    // 7. 测试【分块版本】性能
    cudaEvent_t start_tiled, stop_tiled;
    float time_tiled_ms;
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    printf("🚀 开始运行【分块版本】（共享内存 + __syncthreads()）...\n");
    cudaEventRecord(start_tiled);
    tiledMatrixMultiply<<<gridDim, blockDim>>>(M_d, N_d, P_tiled_d, MATRIX_SIZE);
    cudaEventRecord(stop_tiled);
    cudaEventSynchronize(stop_tiled);
    cudaEventElapsedTime(&time_tiled_ms, start_tiled, stop_tiled);
    printf("⏱️  分块版本耗时: %.2f ms\n\n", time_tiled_ms);

    // 8. 拷贝结果回主机并验证
    CHECK_CUDA_ERROR(cudaMemcpy(P_simple_h, P_simple_d, matrix_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(P_tiled_h, P_tiled_d, matrix_bytes, cudaMemcpyDeviceToHost));
    verifyResult(P_simple_h, P_tiled_h, MATRIX_SIZE);

    // 9. 打印最终性能对比
    printf("=========================================\n");
    printf("📊 性能对比总结\n");
    printf("=========================================\n");
    printf("简单版本: %.2f ms\n", time_simple_ms);
    printf("分块版本: %.2f ms\n", time_tiled_ms);
    printf("🚀 加速比: %.2fx\n", time_simple_ms / time_tiled_ms);
    printf("=========================================\n");

    // 10. 清理资源
    cudaEventDestroy(start_simple);
    cudaEventDestroy(stop_simple);
    cudaEventDestroy(start_tiled);
    cudaEventDestroy(stop_tiled);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_simple_d);
    cudaFree(P_tiled_d);
    free(M_h);
    free(N_h);
    free(P_simple_h);
    free(P_tiled_h);

    return 0;
}