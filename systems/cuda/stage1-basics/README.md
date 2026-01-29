# Stage 1: 新手阶段 - CUDA 基础

## 阶段目标

掌握 CUDA 编程的基本概念和常见算法，能够编写简单但正确的 CUDA 程序。

## 学习内容

### 01. Block Reduce - 块内规约

**核心概念**：
- 并行规约（Parallel Reduction）的思想
- 线程同步 `__syncthreads()`
- 全局内存 vs 共享内存

**实现要求**：
- [ ] 实现基础的邻居配对规约
- [ ] 实现交错配对规约（性能更好）
- [ ] 对比不同实现的性能
- [ ] 理解 warp divergence 的影响

**学习产出**：
- 代码：`01-block-reduce/block_reduce.cu`
- 笔记：理解为什么需要 `__syncthreads()`
- 思考：为什么交错配对比邻居配对快？

**参考资料**：
```
初始数组: [1, 2, 3, 4, 5, 6, 7, 8]

邻居配对 (Neighbored Pairing):
步骤1: [1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]
步骤2: [3+7, 11+15] = [10, 26]
步骤3: [10+26] = [36]

交错配对 (Interleaved Pairing):
步骤1: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
步骤2: [6+10, 8+12] = [16, 20]
步骤3: [16+20] = [36]
```

---

### 02. Matrix Transpose - 矩阵转置

**核心概念**：
- 全局内存访问模式（Global Memory Access Pattern）
- 合并访问（Coalesced Access）
- 跨步访问（Strided Access）的性能问题

**实现要求**：
- [ ] 实现 naive 版本（读写都跨步）
- [ ] 实现 coalesced 读版本
- [ ] 实现 coalesced 写版本
- [ ] 对比三个版本的带宽利用率

**学习产出**：
- 代码：`02-matrix-transpose/matrix_transpose.cu`
- 笔记：理解为什么合并访问如此重要
- 性能对比：三个版本的带宽差异

**示例**：
```
输入矩阵 A (4x4):
1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16

转置后 A^T:
1  5  9  13
2  6  10 14
3  7  11 15
4  8  12 16
```

---

### 03. Simple MatMul - 简单矩阵乘法

**核心概念**：
- 计算密集型操作
- 线程块和网格的组织
- 基础的矩阵乘法算法

**实现要求**：
- [ ] 实现基础的矩阵乘法（C = A × B）
- [ ] 每个线程计算一个输出元素
- [ ] 测量 FLOPS（浮点运算次数/秒）
- [ ] 与 CPU 版本对比性能

**学习产出**：
- 代码：`03-simple-matmul/matmul_naive.cu`
- 笔记：理解矩阵乘法的计算量和访存量
- 性能分析：计算 Compute/Memory Ratio

**算法**：
```
C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]

示例 (2x2 矩阵):
A = [1 2]    B = [5 6]
    [3 4]        [7 8]

C = [1×5+2×7  1×6+2×8] = [19 22]
    [3×5+4×7  3×6+4×8]   [43 50]
```

---

## 通用工具 (common/)

为了方便开发，创建一些通用的工具函数：

```cpp
// common/cuda_utils.h
- CHECK() 宏：CUDA 错误检查
- cpuTimer: CPU 计时器
- cudaTimer: CUDA 事件计时器
- initMatrix(): 矩阵初始化
- compareArrays(): 结果验证
```

---

## 学习检查清单

完成这个阶段后，你应该能够：

- [ ] 解释什么是线程（thread）、线程块（block）、网格（grid）
- [ ] 理解为什么需要 `__syncthreads()`
- [ ] 说出合并访问（coalesced access）的重要性
- [ ] 计算简单 kernel 的带宽利用率
- [ ] 使用 `cudaEvent` 进行性能测试
- [ ] 独立编写和调试简单的 CUDA 程序

---

## 下一步

完成这个阶段后，你已经掌握了 CUDA 编程的基础。

接下来在 **Stage 2** 中，你将学习如何使用 **shared memory** 来优化这些算法，性能可以提升 5-10 倍！

继续加油！🚀
