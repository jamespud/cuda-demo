# Stencil Kernel 实现指南

本文档提供详细的实现指导，帮助你完成各个版本的 stencil kernel。

## 1. CPU 实现 (`cpu_stencil.cpp`)

### 完整实现示例

```cpp
void cpu_stencil(const float* in, float* out, int nx, int ny, int nz) {
    for (int k = 1; k < nz - 1; k++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                int idx = i + j * nx + k * nx * ny;

                out[idx] = kCenter * in[idx] +
                           kNeighbor * (in[idx - 1] + in[idx + 1] +
                                       in[idx - nx] + in[idx + nx] +
                                       in[idx - nx * ny] + in[idx + nx * ny]);
            }
        }
    }
}
```

### 关键点

1. **边界处理**: 循环从 1 开始，到 `nz-1` 结束，跳过所有边界点
2. **索引计算**: 使用 row-major 顺序 `i + j*nx + k*nx*ny`
3. **邻居访问**: 6 个邻居分别是 ±1, ±nx, ±nx*ny

---

## 2. Naive CUDA Kernel (`naive_stencil.cu`)

### Kernel 实现

```cpp
__global__ void naive_stencil_kernel(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1) {
        int idx = i + j * nx + k * nx * ny;

        out[idx] = kCenter * in[idx] +
                   kNeighbor * (in[idx - 1] + in[idx + 1] +
                               in[idx - nx] + in[idx + nx] +
                               in[idx - nx * ny] + in[idx + nx * ny]);
    }
}
```

### Launch 函数

```cpp
void launch_naive(const float* d_in, float* d_out, int nx, int ny, int nz) {
    dim3 block(16, 16, 1);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    naive_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaGetLastError());
}
```

### 关键点

1. **Block 大小**: 16x16x1 是 2D tile 的常用大小
2. **Grid 计算**: 使用 ceiling division `(n + block - 1) / block`
3. **边界检查**: 在 kernel 内部使用 `if` 条件
4. **`__restrict__`**: 告诉编译器指针不重叠，可以优化

---

## 3. Shared Memory Kernel (`shared_stencil.cu`)

### Kernel 实现（2D tile 版本）

```cpp
__global__ void shared_stencil_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int nx, int ny, int nz, int k_offset) {
    constexpr int TILE_SIZE = 16;
    __shared__ float s_data[TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * TILE_SIZE + tx;
    int j = blockIdx.y * TILE_SIZE + ty;

    // Load data into shared memory
    int s_i = tx + 1;  // +1 for halo
    int s_j = ty + 1;

    // Load interior point
    if (i < nx && j < ny) {
        int idx = i + j * nx + k_offset * nx * ny;
        s_data[s_j][s_i] = in[idx];
    }

    // Load halo regions (simplified - need boundary checks)
    if (tx == 0 && i > 0) {
        int idx = (i - 1) + j * nx + k_offset * nx * ny;
        s_data[s_j][0] = in[idx];
    }
    // ... load other halo regions

    __syncthreads();

    // Compute stencil
    if (tx > 0 && tx < TILE_SIZE + 1 && ty > 0 && ty < TILE_SIZE + 1) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            int idx = i + j * nx + k_offset * nx * ny;

            out[idx] = kCenter * s_data[s_j][s_i] +
                       kNeighbor * (s_data[s_j][s_i - 1] + s_data[s_j][s_i + 1] +
                                   s_data[s_j - 1][s_i] + s_data[s_j + 1][s_i]);
            // + z-direction neighbors from global memory
        }
    }
}
```

### Launch 函数

```cpp
void launch_shared(const float* d_in, float* d_out, int nx, int ny, int nz) {
    constexpr int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((nx + TILE_SIZE - 1) / TILE_SIZE,
              (ny + TILE_SIZE - 1) / TILE_SIZE,
              1);

    for (int k = 1; k < nz - 1; k++) {
        int k_offset = k * nx * ny;
        shared_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz, k_offset);
        CHECK_CUDA(cudaGetLastError());
    }
}
```

### 关键点

1. **Halo**: 共享内存需要额外 2 个元素（每个方向 +1）
2. **加载策略**: 每个线程加载一个元素，边界线程加载 halo
3. **同步**: `__syncthreads()` 确保所有数据加载完成
4. **Z 方向**: 可以逐个 plane 处理，或使用 3D 共享内存

---

## 4. Coarsened Kernel (`coarsened_stencil.cu`)

### Kernel 实现

```cpp
__global__ void coarsened_stencil_kernel(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int nx, int ny, int nz) {
    constexpr int TILE_SIZE = 16;
    constexpr int COARSENING_FACTOR = 4;

    __shared__ float s_data[COARSENING_FACTOR + 2][TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.x * TILE_SIZE + tx;
    int j = blockIdx.y * TILE_SIZE + ty;
    int k_base = blockIdx.z * COARSENING_FACTOR;

    // Load multiple z-planes into shared memory
    for (int c = 0; c < COARSENING_FACTOR + 2; c++) {
        int k = k_base + c - 1;  // -1 for halo

        if (k >= 0 && k < nz && i < nx && j < ny) {
            int idx = i + j * nx + k * nx * ny;
            s_data[c][ty + 1][tx + 1] = in[idx];
        }
    }

    __syncthreads();

    // Compute stencil with coarsening
    if (tx > 0 && tx < TILE_SIZE + 1 && ty > 0 && ty < TILE_SIZE + 1) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            for (int c = 0; c < COARSENING_FACTOR; c++) {
                int k = k_base + c;

                if (k > 0 && k < nz - 1) {
                    int idx = i + j * nx + k * nx * ny;

                    out[idx] = kCenter * s_data[c + 1][ty + 1][tx + 1] +
                               kNeighbor * (s_data[c + 1][ty + 1][tx] +
                                           s_data[c + 1][ty + 1][tx + 2] +
                                           s_data[c + 1][ty][tx + 1] +
                                           s_data[c + 1][ty + 2][tx + 1] +
                                           s_data[c][ty + 1][tx + 1] +
                                           s_data[c + 2][ty + 1][tx + 1]);
                }
            }
        }
    }
}
```

### Launch 函数

```cpp
void launch_coarsened(const float* d_in, float* d_out, int nx, int ny, int nz) {
    constexpr int TILE_SIZE = 16;
    constexpr int COARSENING_FACTOR = 4;

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((nx + TILE_SIZE - 1) / TILE_SIZE,
              (ny + TILE_SIZE - 1) / TILE_SIZE,
              (nz + COARSENING_FACTOR - 1) / COARSENING_FACTOR);

    coarsened_stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
    CHECK_CUDA(cudaGetLastError());
}
```

### 关键点

1. **Coarsening**: 每个线程计算多个 z-planes
2. **共享内存**: 3D 数组存储多个 planes
3. **Halo**: z 方向也需要 halo（+2）
4. **循环展开**: 内层循环计算多个输出
5. **Grid 调整**: z 方向 grid 减少为原来的 1/COARSENING_FACTOR

---

## 优化技巧总结

### 内存访问优化

1. **Coalesced Access**: 确保线程访问连续的内存地址
2. **Shared Memory**: 减少全局内存访问
3. **Cache Blocking**: 使用 tile 策略提高缓存命中率

### 计算优化

1. **Loop Unrolling**: 编译器自动或手动展开循环
2. **Instruction Level Parallelism**: Coarsening 提高 ILP
3. **Register Usage**: 平衡寄存器使用和 occupancy

### 边界处理

1. **Halo Loading**: 共享内存需要加载边界数据
2. **Conditionals**: 最小化分支 divergence
3. **Padding**: 可以考虑 padding 简化边界处理

---

## 调试建议

1. **小规模测试**: 先用小网格（如 16x16x16）测试
2. **可视化输出**: 打印部分结果验证正确性
3. **CUDA-GDB**: 使用调试器定位问题
4. **Nsight Compute**: 分析性能瓶颈

---

## 性能分析工具

```bash
# 使用 Nsight Compute 分析 kernel
ncu --set full ./stencil

# 使用 Nsight Systems 分析整体性能
nsys profile --stats=true ./stencil
```

---

## 常见问题

### Q: 为什么 shared memory 版本比 naive 慢？
A: 可能是：
- 共享内存使用不当（bank conflicts）
- Halo 加载开销太大
- Block 大小不合适

### Q: 如何选择 coarsening factor？
A: 取决于：
- 寄存器数量（每个线程）
- 共享内存大小
- 具体硬件架构
- 通常 4-8 是合理的范围

### Q: 如何处理边界条件？
A: 有几种策略：
- 在 kernel 内部检查边界
- 使用 padding（填充边界）
- 单独处理边界区域
