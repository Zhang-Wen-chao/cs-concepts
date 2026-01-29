# CUDA 学习路线图

这是一个从零到精通的 CUDA 编程学习路径，旨在帮助你成为一名优秀的 GPU 编程工程师。

## 学习目标

完成本路线后，你将能够：
- 熟练编写高性能 CUDA 程序
- 优化内存访问模式，避免性能瓶颈
- 实现深度学习中的核心算子
- 使用专业工具进行性能分析和调优
- 找到一份优秀的 GPU 编程相关工作

## 学习路线

### Stage 1: 新手阶段 - CUDA 基础 (stage1-basics/)

**目标**：掌握 CUDA 编程基本概念和常见算法

- [x] **01-block-reduce** - 块内规约算法
  - 理解并行规约的思想
  - 学会使用 `__syncthreads()` 同步
  - 实现基础的求和规约

- [ ] **02-matrix-transpose** - 矩阵转置
  - 理解全局内存访问模式
  - 实现高效的矩阵转置
  - 对比 naive 和 coalesced 访问的性能差异

- [ ] **03-simple-matmul** - 简单矩阵乘法
  - 实现基础的矩阵乘法算法
  - 理解计算密集型操作的特点
  - 建立性能分析的基础

### Stage 2: 进阶阶段 - 内存优化 (stage2-intermediate/)

**目标**：深入理解 CUDA 内存层次结构，掌握性能优化技巧

- [ ] **01-shared-memory-optimization** - 共享内存优化
  - 将 Stage 1 的算法改写为 shared memory 版本
  - 理解 shared memory 的优势
  - 对比性能提升

- [ ] **02-bank-conflict-avoidance** - 规避 Bank 冲突
  - 理解 shared memory 的 bank 结构
  - 识别和解决 bank conflict
  - 实现 padding 等优化技巧

- [ ] **03-radix-sort** - CUDA 基数排序
  - 前缀和 (Prefix Sum / Scan)
  - 排他前缀和 (Exclusive Scan)
  - 直方图 (Histogram)
  - 基数排序完整实现

- [ ] **04-cub-library-study** - CUB 库源码学习
  - 学习 NVIDIA CUB 库的实现
  - 理解高性能并行原语的设计
  - **这一步完成后，你已经领先一大票人了！**

### Stage 3: 高阶阶段 - 深度学习算子 (stage3-advanced/)

**目标**：手写深度学习核心算子，达到工业级水平

- [ ] **01-forward-backward-propagation** - 前向反向传播
  - 手写全连接层的前向传播
  - 手写反向传播和梯度计算
  - 理解自动微分的 CUDA 实现

- [ ] **02-convolution** - 卷积算子
  - 实现 2D 卷积
  - Im2Col + GEMM 方法
  - Winograd 快速卷积

- [ ] **03-flash-attention** - FlashAttention
  - 理解 Attention 的计算瓶颈
  - 实现 FlashAttention v1/v2
  - IO-aware 算法设计

- [ ] **04-high-performance-matmul** - 高性能矩阵乘法
  - Tiling 技术
  - 寄存器 blocking
  - Warp-level 优化
  - 达到 cuBLAS 80%+ 性能

### Stage 4: 专家阶段 - 性能分析与调优 (stage4-expert/)

**目标**：掌握专业性能分析工具，成为调优专家

- [ ] **01-nsight-systems** - Nsight Systems 系统级分析
  - Timeline 分析
  - Kernel 启动开销
  - CPU-GPU 协同优化

- [ ] **02-nsight-compute** - Nsight Compute 内核级分析
  - Roofline 模型
  - Memory/Compute bound 识别
  - 指令级优化

- [ ] **03-profiling-optimization** - 综合性能调优
  - 真实项目性能调优案例
  - 从分析到优化的完整流程

## 学习建议

1. **循序渐进**：按照 Stage 1 → Stage 2 → Stage 3 → Stage 4 的顺序学习
2. **动手实践**：每个算法都要自己写一遍，不要只看代码
3. **性能对比**：每次优化都要测量性能提升，建立量化思维
4. **阅读源码**：学习 CUB、cutlass、cuBLAS 等优秀库的实现
5. **工具使用**：熟练使用 Nsight 系列工具是必备技能

## 学习资源

### 书籍
- 《CUDA C Programming Guide》- NVIDIA 官方文档
- 《Programming Massively Parallel Processors》
- 《CUDA by Example》

### 在线资源
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA CUB Library](https://github.com/NVIDIA/cub)
- [cutlass - CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)

### 论文
- FlashAttention: Fast and Memory-Efficient Exact Attention
- Faster Neural Networks with Cutlass
- cuDNN: Efficient Primitives for Deep Learning

## 进度追踪

使用下面的 checklist 追踪你的学习进度：

```
Stage 1 (新手): □ 0/3
  □ Block Reduce
  □ Matrix Transpose
  □ Simple MatMul

Stage 2 (进阶): □ 0/4
  □ Shared Memory Optimization
  □ Bank Conflict Avoidance
  □ Radix Sort (Scan + Histogram)
  □ CUB Library Study

Stage 3 (高阶): □ 0/4
  □ Forward/Backward Propagation
  □ Convolution
  □ Flash Attention
  □ High-Performance MatMul

Stage 4 (专家): □ 0/3
  □ Nsight Systems
  □ Nsight Compute
  □ Profiling & Optimization
```

## 完成标准

每个项目完成的标准：
1. ✅ 代码能正确运行
2. ✅ 有详细的注释说明
3. ✅ 有性能测试和对比
4. ✅ 写了学习笔记总结
5. ✅ （可选）写了博客文章分享

---

**要是能完成高阶内容，基本上找份很好的工作没问题！** 💪

加油！🚀
