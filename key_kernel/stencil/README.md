# 3D Stencil Kernel Optimization Framework

这是一个完整的 3D 7-point stencil kernel 优化框架，从 CPU 基准实现到 CUDA 逐步优化。

## 项目结构

```
stencil/
├── CMakeLists.txt              # CMake 构建配置
├── README.md                   # 本文件
├── include/
│   ├── common.cuh             # 通用工具函数和宏定义
│   └── stencil.cuh            # Stencil kernel 接口声明
└── src/
    ├── main.cu                # 主程序和性能测试
    ├── cpu_stencil.cpp        # CPU 基准实现
    ├── naive_stencil.cu       # Naive CUDA kernel
    ├── shared_stencil.cu      # Shared memory 优化
    └── coarsened_stencil.cu   # Coarsening 优化
```

## 构建和运行

```bash
cd /mnt/d/proj/cuda-demo/key_kernel/stencil
mkdir -p build && cd build
cmake ..
make
./stencil
```

## 优化路径

### 1. CPU 基准实现 (`cpu_stencil.cpp`)

**目标**: 提供正确的参考结果

**TODO**:
- 实现 7-point 3D stencil 公式
- 处理边界条件（跳过边界点）
- 使用 row-major 索引计算

**公式**:
```
out[idx] = kCenter * in[idx] +
           kNeighbor * (in[idx-1] + in[idx+1] +
                       in[idx-nx] + in[idx+nx] +
                       in[idx-nx*ny] + in[idx+nx*ny])
```

### 2. Naive CUDA Kernel (`naive_stencil.cu`)

**目标**: 基本的 CUDA 实现，每个线程计算一个输出元素

**TODO**:
- 配置 block 和 grid 维度
- 计算全局线程索引 (i, j, k)
- 检查边界条件
- 应用 stencil 公式
- 写入全局内存

**优化点**:
- 使用 `__restrict__` 提示编译器优化
- 合理的 block 大小（如 16x16x1）

### 3. Shared Memory Kernel (`shared_stencil.cu`)

**目标**: 使用共享内存减少全局内存访问

**TODO**:
- 声明带 halo 的共享内存数组
- 加载数据到共享内存（包括 halo 区域）
- 处理边界条件和 halo 加载
- 同步线程 (`__syncthreads()`)
- 使用共享内存计算 stencil
- 写入全局内存

**优化点**:
- 共享内存大小: `(TILE_SIZE + 2) x (TILE_SIZE + 2)`
- 每个 z-plane 单独处理或使用 3D 共享内存
- 减少全局内存访问次数

### 4. Coarsened Kernel (`coarsened_stencil.cu`)

**目标**: 结合共享内存和循环展开，提高指令级并行

**TODO**:
- 声明多维共享内存数组（包含多个 z-planes）
- 加载多个 z-planes 到共享内存
- 每个线程计算多个输出元素（coarsening factor）
- 循环展开计算多个 z-planes
- 优化内存访问模式

**优化点**:
- Coarsening factor: 每个线程计算 4 个 z-planes
- 共享内存: `(COARSENING_FACTOR + 2) x (TILE_SIZE + 2) x (TILE_SIZE + 2)`
- 提高指令级并行性
- 更好的缓存利用率

## 性能指标

程序会输出以下性能指标：

1. **CPU 基准时间**: 参考实现的时间
2. **Naive CUDA 时间**: 基本实现的时间和加速比
3. **Shared Memory 时间**: 共享内存优化的时间和加速比
4. **Coarsened 时间**: 最终优化的时间和加速比

每个版本都会验证正确性（与 CPU 结果对比）。

## 关键常量

在 `common.cuh` 中定义：

```cpp
constexpr float kCenter = 0.5f;      // 中心点权重
constexpr float kNeighbor = 0.0833f; // 邻居点权重
```

## 实现提示

### CPU 实现
- 使用三层嵌套循环
- 跳过所有边界点（i=0, i=nx-1, j=0, j=ny-1, k=0, k=nz-1）
- 计算线性索引: `idx = i + j * nx + k * nx * ny`

### Naive Kernel
- Block 大小建议: `dim3(16, 16, 1)` 或 `dim3(8, 8, 8)`
- Grid 大小: `(nx + block.x - 1) / block.x`
- 使用 `if` 条件检查边界

### Shared Memory Kernel
- 共享内存需要 halo（每个方向多 1 个元素）
- 需要处理边界加载（可能需要条件判断）
- 使用 `__syncthreads()` 同步
- 可以逐个 z-plane 处理

### Coarsened Kernel
- 每个线程处理多个 z-planes
- 共享内存需要存储多个 z-planes
- 循环展开提高并行性
- 注意 z 方向的边界检查

## 预期性能提升

- **Naive vs CPU**: 10-50x 加速（取决于硬件）
- **Shared vs Naive**: 2-4x 加速（减少内存访问）
- **Coarsened vs Shared**: 1.5-2x 加速（提高指令并行）

## 注意事项

1. 所有 kernel 实现都标记为 `TODO`，需要手动完成
2. 框架提供了完整的性能测试和验证逻辑
3. 每个版本都会与 CPU 结果对比验证正确性
4. 使用 CUDA Events 进行精确计时
5. 错误检查使用 `CHECK_CUDA` 宏

## 扩展建议

完成基本实现后，可以考虑：

1. 使用 CUDA Streams 实现异步执行
2. 使用 Unified Memory (cudaMallocManaged)
3. 实现更高阶的 stencil（如 27-point）
4. 使用纹理内存优化
5. 实现多 GPU 版本
