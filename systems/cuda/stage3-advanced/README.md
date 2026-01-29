# Stage 3: 高阶阶段 - 深度学习算子

## 阶段目标

手写深度学习核心算子，掌握工业级高性能计算技能，达到可以面试顶级 AI 公司的水平。

## 为什么要手写深度学习算子？

- **工业需求**：AI 公司（如 NVIDIA、Meta、Google、OpenAI）都需要算子优化工程师
- **性能关键**：模型训练/推理的性能瓶颈往往在算子上
- **高薪岗位**：掌握这些技能可以拿到非常好的 offer
- **深入理解**：只有自己写过，才能真正理解框架底层的实现

---

## 学习内容

### 01. Forward & Backward Propagation - 前向反向传播

**核心概念**：
- 深度学习的自动微分（Autograd）
- 前向传播的 CUDA 实现
- 反向传播的梯度计算
- 计算图和内存管理

**实现要求**：
- [ ] 实现全连接层的前向传播
  ```
  Y = X @ W + b
  ```
- [ ] 实现常见激活函数（ReLU, Sigmoid, Tanh）
- [ ] 实现反向传播计算梯度
  ```
  dL/dW = X^T @ dL/dY
  dL/dX = dL/dY @ W^T
  dL/db = sum(dL/dY)
  ```
- [ ] 实现简单的两层神经网络并训练

**学习产出**：
- 代码：`01-forward-backward-propagation/`
  - `linear_forward.cu` - 线性层前向
  - `activations.cu` - 激活函数
  - `linear_backward.cu` - 线性层反向
  - `simple_mlp.cu` - 完整的两层网络
- 笔记：理解自动微分的原理
- 实验：在 MNIST 上训练并验证正确性

---

### 02. Convolution - 卷积算子

**核心概念**：
- 2D 卷积的数学原理
- Im2Col + GEMM 方法
- Winograd 快速卷积
- cuDNN 的实现策略

**实现要求**：
- [ ] 实现 Naive 卷积（7 层循环）
- [ ] 实现 Im2Col + GEMM 卷积
- [ ] （可选）实现 Winograd F(2x2, 3x3) 卷积
- [ ] 对比三种方法的性能
- [ ] 支持 padding、stride、dilation

**学习产出**：
- 代码：`02-convolution/`
  - `conv_naive.cu` - 朴素实现
  - `im2col.cu` - Im2Col 变换
  - `conv_gemm.cu` - Im2Col + GEMM
  - `conv_winograd.cu` - Winograd（可选）
- 笔记：理解为什么 Im2Col + GEMM 更快
- 性能测试：与 cuDNN 对比

**卷积示例**：
```
输入 (4x4):           卷积核 (3x3):
1  2  3  4           1  0  -1
5  6  7  8           1  0  -1
9  10 11 12          1  0  -1
13 14 15 16

输出 (2x2):
-12  -12
-12  -12
```

**Im2Col 思想**：
```
将卷积转换为矩阵乘法：
Conv(X, W) = Im2Col(X) @ Flatten(W)

优势：可以复用高度优化的 GEMM 实现
```

---

### 03. Flash Attention - 高效注意力机制

**核心概念**：
- Attention 的计算瓶颈分析
- IO-aware 算法设计
- Tiling 和 Recomputation 策略
- FlashAttention v1 和 v2 的区别

**实现要求**：
- [ ] 实现标准 Attention
  ```
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
  ```
- [ ] 分析 IO 复杂度和内存占用
- [ ] 实现 FlashAttention v1（Tiling + Online Softmax）
- [ ] （可选）实现 FlashAttention v2（优化 work partitioning）
- [ ] 验证数值稳定性和正确性

**学习产出**：
- 代码：`03-flash-attention/`
  - `attention_naive.cu` - 标准实现
  - `flash_attention_v1.cu` - FlashAttention v1
  - `flash_attention_v2.cu` - FlashAttention v2（可选）
- 笔记：理解 IO-aware 算法的设计思想
- 论文笔记：精读 FlashAttention 论文
- 性能对比：内存占用和速度提升

**为什么 Flash Attention 重要？**
- Transformer 的核心组件
- 标准实现的内存复杂度是 O(N²)
- FlashAttention 可以减少到 O(N)
- 2-4x 加速，且支持更长的序列

**参考论文**：
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

---

### 04. High-Performance MatMul - 高性能矩阵乘法

**核心概念**：
- Tiling（分块）策略
- 寄存器 blocking
- Warp-level GEMM
- Tensor Core 的使用

**实现要求**：
- [ ] 实现多级 Tiling（Global → Shared → Register）
- [ ] 实现 Warp-level GEMM（使用 WMMA API）
- [ ] 实现 double buffering 隐藏访存延迟
- [ ] 优化到达 cuBLAS 80%+ 性能
- [ ] 支持不同数据类型（FP32、FP16、INT8）

**学习产出**：
- 代码：`04-high-performance-matmul/`
  - `matmul_tiled.cu` - 基础 Tiling
  - `matmul_register_blocking.cu` - 寄存器优化
  - `matmul_wmma.cu` - Tensor Core 版本
  - `matmul_optimized.cu` - 最终优化版本
- 笔记：理解现代 GPU 的内存层次
- Roofline 分析：计算和访存的平衡

**优化路径**：
```
Naive (50 GFLOPS)
  ↓ Shared Memory Tiling
Tiled (500 GFLOPS)
  ↓ Register Blocking
Register (2 TFLOPS)
  ↓ Warp-level GEMM
WMMA (5 TFLOPS)
  ↓ Double Buffering + 其他优化
Final (8 TFLOPS, ~80% of cuBLAS)
```

**性能目标（V100 GPU）**：
- FP32: > 8 TFLOPS (cuBLAS ~10 TFLOPS)
- FP16 (Tensor Core): > 60 TFLOPS (cuBLAS ~80 TFLOPS)

**参考资源**：
- [cutlass - CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)

---

## 学习检查清单

完成这个阶段后，你应该能够：

- [ ] 手写深度学习的前向和反向传播
- [ ] 实现高性能卷积算子
- [ ] 理解和实现 FlashAttention
- [ ] 写出接近 cuBLAS 性能的矩阵乘法
- [ ] 分析算子的计算和 IO 复杂度
- [ ] 使用 Tensor Core 加速计算
- [ ] 阅读和理解顶会论文中的算法

---

## 性能目标

| 算子 | 目标性能 (V100) |
|------|----------------|
| Linear Forward/Backward | 80% of cuBLAS |
| Convolution | 70% of cuDNN |
| Flash Attention | 2-4x faster than naive |
| MatMul (FP32) | > 8 TFLOPS |
| MatMul (FP16 Tensor Core) | > 60 TFLOPS |

---

## 项目建议

完成这些算子后，可以尝试：

1. **集成到 PyTorch**：写 PyTorch Custom Operator
2. **端到端训练**：用自己的算子训练一个模型
3. **性能对比**：与 PyTorch/cuDNN 详细对比
4. **写博客**：分享你的实现和优化心得

---

## 面试准备

这个阶段的内容是面试重点：

- **NVIDIA**：会问 GEMM 优化、Tensor Core 使用
- **Meta/Google**：会问卷积和 Attention 的优化
- **OpenAI**：会问 Flash Attention 的原理
- **国内大厂**：会考察实际手写能力

准备好代码和笔记，可以在面试中展示！

---

## 下一步

完成这个阶段后，你已经具备了世界级的 GPU 编程能力！

接下来在 **Stage 4** 中，你将学习使用专业工具进行性能分析和调优，这是从优秀到卓越的最后一步！

**完成这个阶段，找份很好的工作没问题！** 💪🚀
