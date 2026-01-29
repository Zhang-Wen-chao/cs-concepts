# Stage 2: 进阶阶段 - 内存优化

## 阶段目标

深入理解 CUDA 内存层次结构，掌握高性能内存优化技巧，学习并行算法的工业级实现。

## 为什么要学习内存优化？

**性能瓶颈通常在内存，而非计算！**

- GPU 的计算能力远超内存带宽
- 优化内存访问可以带来 5-10x 甚至更高的性能提升
- 这是初级和高级 CUDA 程序员的分水岭

---

## 学习内容

### 01. Shared Memory Optimization - 共享内存优化

**核心概念**：
- Shared Memory 的特点和优势
- On-chip memory vs Off-chip memory
- Shared Memory 的生命周期

**实现要求**：
- [ ] 将 Stage 1 的 Block Reduce 改写为 Shared Memory 版本
- [ ] 将 Stage 1 的 Matrix Transpose 改写为 Shared Memory 版本
- [ ] 将 Stage 1 的 MatMul 改写为 Shared Memory 版本（Tiling）
- [ ] 对比性能提升（应该有 5-10x）

**学习产出**：
- 代码：三个优化版本的实现
- 笔记：Shared Memory 的使用场景和限制
- 性能对比图表

**关键点**：
```cuda
// Shared Memory 声明
__shared__ float sharedData[BLOCK_SIZE];

// 从 Global Memory 加载到 Shared Memory
sharedData[tid] = globalData[tid];
__syncthreads();  // 必须同步！

// 在 Shared Memory 上计算
result = sharedData[tid] + sharedData[tid + 1];
```

---

### 02. Bank Conflict Avoidance - 规避 Bank 冲突

**核心概念**：
- Shared Memory 的 Bank 结构
- Bank Conflict 的性能影响
- Padding 和其他规避技巧

**实现要求**：
- [ ] 理解 32-way banked shared memory
- [ ] 识别并测量 bank conflict
- [ ] 实现 padding 优化矩阵转置
- [ ] 对比有/无 bank conflict 的性能

**学习产出**：
- 代码：`02-bank-conflict-avoidance/transpose_no_conflict.cu`
- 笔记：Bank conflict 的原理和解决方案
- 性能分析：使用 Nsight Compute 查看 bank conflict 指标

**Bank Conflict 示例**：
```
32 个线程访问 Shared Memory:

有冲突 (stride = 2):
Thread 0 -> Bank 0
Thread 1 -> Bank 2
Thread 2 -> Bank 4
...
Thread 16 -> Bank 0 (冲突!)

无冲突 (stride = 1):
Thread 0 -> Bank 0
Thread 1 -> Bank 1
Thread 2 -> Bank 2
...
Thread 31 -> Bank 31
```

---

### 03. Radix Sort - CUDA 基数排序

**核心概念**：
- 前缀和（Prefix Sum / Scan）
- 排他前缀和（Exclusive Scan）
- 直方图（Histogram）
- 基数排序的完整实现

**实现要求**：
- [ ] 实现 Inclusive Scan（包含前缀和）
- [ ] 实现 Exclusive Scan（排他前缀和）
- [ ] 实现 Histogram（直方图）
- [ ] 组合以上实现完整的 Radix Sort
- [ ] 支持 Block-level 和 Device-level

**学习产出**：
- 代码：`03-radix-sort/`
  - `scan.cu` - 前缀和实现
  - `histogram.cu` - 直方图实现
  - `radix_sort.cu` - 基数排序完整实现
- 笔记：理解 Scan 算法的并行化
- 性能测试：与 `thrust::sort` 对比

**算法示例**：
```
Inclusive Scan (累加和):
输入: [3, 1, 7, 0, 4, 1, 6, 3]
输出: [3, 4, 11, 11, 15, 16, 22, 25]

Exclusive Scan (前缀和):
输入: [3, 1, 7, 0, 4, 1, 6, 3]
输出: [0, 3, 4, 11, 11, 15, 16, 22]

Histogram (统计每个值的位置):
输入: [3, 1, 7, 0, 4, 1, 6, 3]
→ 按最低位排序 → [0, 1, 1, 3, 3, 4, 6, 7]
```

---

### 04. CUB Library Study - CUB 库源码学习

**核心概念**：
- NVIDIA CUB（CUDA UnBound）库
- 高性能并行原语的设计模式
- Warp-level、Block-level、Device-level 抽象

**学习要求**：
- [ ] 安装并编译 CUB 库
- [ ] 阅读 `BlockReduce` 源码
- [ ] 阅读 `BlockScan` 源码
- [ ] 阅读 `DeviceRadixSort` 源码
- [ ] 理解其中的优化技巧

**学习产出**：
- 笔记：`04-cub-library-study/cub_notes.md`
  - CUB 的架构设计
  - Warp-level primitives
  - Block-level primitives
  - Device-level primitives
- 代码：使用 CUB 重写前面的算法
- 对比：自己实现 vs CUB 的性能差距

**CUB 示例**：
```cuda
#include <cub/cub.cuh>

// Block-level reduce
typedef cub::BlockReduce<int, 256> BlockReduce;
__shared__ typename BlockReduce::TempStorage temp_storage;

int thread_data = ...;
int aggregate = BlockReduce(temp_storage).Sum(thread_data);

// Device-level radix sort
int *d_keys;
cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                d_keys, num_items);
```

**重要性**：
- CUB 是 NVIDIA 官方的高性能库
- Thrust 底层使用 CUB 实现
- 学习 CUB 可以了解工业级 CUDA 代码的写法
- **掌握 CUB 后，你已经领先一大票 CUDA 程序员了！**

---

## 学习检查清单

完成这个阶段后，你应该能够：

- [ ] 熟练使用 Shared Memory 优化算法
- [ ] 识别和解决 Bank Conflict
- [ ] 实现 Scan、Histogram 等并行原语
- [ ] 理解 Radix Sort 的并行实现
- [ ] 阅读和理解 CUB 库的源码
- [ ] 写出接近工业级性能的 CUDA 代码

---

## 性能目标

在这个阶段结束时，你的实现应该达到：

| 算法 | 目标性能 |
|------|---------|
| Block Reduce | 接近理论峰值 |
| Matrix Transpose | > 500 GB/s (V100) |
| Matrix Multiply | > 2 TFLOPS (V100) |
| Radix Sort | 80% of CUB/Thrust |

---

## 下一步

完成这个阶段后，你已经具备了写高性能 CUDA 代码的能力。

接下来在 **Stage 3** 中，你将挑战深度学习的核心算子，包括卷积、Attention、矩阵乘法等，这些是工业界最看重的技能！

继续加油！💪
