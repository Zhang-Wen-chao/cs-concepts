# Stage 4: 专家阶段 - 性能分析与调优

## 阶段目标

掌握专业性能分析工具，成为 CUDA 性能调优专家，能够系统性地分析和优化 GPU 程序。

## 为什么需要性能分析工具？

- **找到瓶颈**：不测量就是在猜测
- **量化优化**：用数据说话，而非直觉
- **专业能力**：这是区分初级和高级工程师的标志
- **工业标准**：所有大公司都使用这些工具

> "Premature optimization is the root of all evil" - Donald Knuth
>
> 但是，**Profiling is not premature！** 先测量，再优化。

---

## 工具概览

NVIDIA 提供了两个核心性能分析工具：

| 工具 | 作用范围 | 主要功能 |
|------|---------|---------|
| **Nsight Systems** | 系统级 | Timeline 分析、CPU-GPU 协同、多流并发 |
| **Nsight Compute** | Kernel 级 | 指令分析、内存吞吐、Roofline 模型 |

使用原则：
1. 先用 **Nsight Systems** 找到慢的 kernel
2. 再用 **Nsight Compute** 深入分析这个 kernel
3. 优化后重复 1-2

---

## 学习内容

### 01. Nsight Systems - 系统级性能分析

**核心概念**：
- Timeline 可视化
- CPU-GPU 协同分析
- Kernel 启动开销
- CUDA Stream 并发分析
- 数据传输优化

**学习要求**：
- [ ] 安装 Nsight Systems
- [ ] 分析 CUDA 程序的 Timeline
- [ ] 识别 CPU-GPU 同步点
- [ ] 优化 Kernel 启动开销
- [ ] 优化数据传输（H2D/D2H）
- [ ] 分析多流并发执行

**学习产出**：
- 笔记：`01-nsight-systems/nsight_systems_guide.md`
  - Timeline 的解读方法
  - 常见性能问题和解决方案
- 案例：分析 Stage 3 的算子，找到优化点
- 报告：优化前后的对比

**分析流程**：
```bash
# 1. 运行程序并收集数据
nsys profile -o my_profile ./my_cuda_program

# 2. 打开 GUI 查看
nsys-ui my_profile.nsys-rep

# 3. 关注以下指标：
# - Kernel 执行时间
# - Memory Copy 时间
# - CPU-GPU Gap
# - Stream 利用率
```

**常见问题和解决方案**：

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| Kernel 启动开销大 | 很多小 kernel | 合并 kernel 或使用 persistent kernel |
| CPU-GPU 同步多 | Timeline 有间隙 | 使用异步 API，减少 `cudaDeviceSynchronize()` |
| 内存拷贝慢 | H2D/D2H 占比高 | 使用 Pinned Memory、减少拷贝次数 |
| Stream 利用率低 | 只有一个 stream | 使用多 stream 并发执行 |

---

### 02. Nsight Compute - Kernel 级性能分析

**核心概念**：
- Roofline 模型
- Memory Bound vs Compute Bound
- Warp Occupancy 和 Utilization
- 指令级分析
- Bank Conflict 检测

**学习要求**：
- [ ] 安装 Nsight Compute
- [ ] 理解 Roofline 模型
- [ ] 分析 Kernel 的限制因素
- [ ] 使用指标识别优化方向
- [ ] 对比优化前后的性能

**学习产出**：
- 笔记：`02-nsight-compute/nsight_compute_guide.md`
  - Roofline 模型的解读
  - 关键性能指标的含义
  - 优化策略决策树
- 案例：深入分析 MatMul Kernel
- 报告：瓶颈分析和优化建议

**分析流程**：
```bash
# 1. 运行并收集详细数据
ncu -o my_kernel --set full ./my_cuda_program

# 2. 打开 GUI 查看
ncu-ui my_kernel.ncu-rep

# 3. 查看关键 Section：
# - GPU Speed of Light (SOL)
# - Memory Workload Analysis
# - Compute Workload Analysis
# - Occupancy
# - Warp State Statistics
```

**关键性能指标**：

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **SOL Memory** | 内存吞吐占峰值的百分比 | > 80% |
| **SOL Compute** | 计算吞吐占峰值的百分比 | > 80% |
| **Achieved Occupancy** | 实际 Warp 占用率 | > 50% |
| **Memory Throughput** | 实际内存带宽 | 接近峰值 |
| **Shared Memory Bank Conflicts** | 共享内存 Bank 冲突 | = 0 |
| **Branch Efficiency** | 分支效率 | 100% |

**Roofline 模型**：
```
         ^
         |         /
Compute  |        / (Compute Bound)
TFLOPS   |       /
         |------/
         |     /|
         |    / | (Memory Bound)
         |   /  |
         +--+---+--------->
            Arithmetic Intensity (FLOP/Byte)
```

**优化决策树**：
```
分析 Kernel
  │
  ├─ Memory Bound?
  │   ├─ Yes → 优化访存模式
  │   │         - 使用 Shared Memory
  │   │         - 优化 Coalescing
  │   │         - 减少 Global Memory 访问
  │   │
  │   └─ No → Compute Bound?
  │       ├─ Yes → 增加计算强度
  │       │         - 提高 Arithmetic Intensity
  │       │         - 使用 Tensor Core
  │       │         - 循环展开
  │       │
  │       └─ No → Latency Bound?
  │           └─ Yes → 提高并行度
  │                     - 增加 Occupancy
  │                     - 减少寄存器使用
  │                     - 优化 Warp 调度
```

---

### 03. Profiling & Optimization - 综合性能调优

**核心概念**：
- 端到端的性能优化流程
- 从分析到优化的系统方法
- 真实项目的性能调优案例

**实践项目**：
- [ ] 选择一个 Stage 3 的算子进行深度优化
- [ ] 完整的分析 → 优化 → 验证流程
- [ ] 达到或超过工业级库的性能

**学习产出**：
- 代码：优化后的完整实现
- 笔记：`03-profiling-optimization/optimization_case_study.md`
  - 初始性能基线
  - 瓶颈分析
  - 优化策略
  - 每次优化的效果
  - 最终性能和结论
- 报告：可以在面试中展示的案例

**优化案例：高性能 MatMul**

```
迭代 1: Naive 实现
  Nsight Systems: Kernel 时间 100ms
  Nsight Compute: Memory Bound, SOL Memory 30%
  → 优化：使用 Shared Memory Tiling

迭代 2: Shared Memory Tiling
  Kernel 时间: 20ms (5x 提升)
  Nsight Compute: 仍然 Memory Bound, Bank Conflict 检测到
  → 优化：Padding 解决 Bank Conflict

迭代 3: 解决 Bank Conflict
  Kernel 时间: 15ms (6.7x 提升)
  Nsight Compute: 开始 Compute Bound, Occupancy 只有 30%
  → 优化：调整 Block Size，增加 Occupancy

迭代 4: 优化 Occupancy
  Kernel 时间: 10ms (10x 提升)
  Nsight Compute: Compute Bound, 但 IPC 较低
  → 优化：Register Blocking + 指令级并行

迭代 5: Register Blocking
  Kernel 时间: 5ms (20x 提升)
  Nsight Compute: 接近 Roofline 上限
  → 优化：使用 Tensor Core (WMMA)

迭代 6: Tensor Core
  Kernel 时间: 1.5ms (67x 提升)
  Nsight Compute: SOL Compute 85%
  → 性能已达到预期，优化完成！
```

---

## 性能优化最佳实践

### 通用原则

1. **先测量，再优化**
   - 不要凭感觉优化
   - 用 profiler 找到真正的瓶颈

2. **渐进式优化**
   - 每次只改一个地方
   - 测量每次改动的效果

3. **建立 Baseline**
   - 记录初始性能
   - 与工业级库对比

4. **量化目标**
   - 设定明确的性能目标（如达到 cuBLAS 80%）
   - 知道何时停止优化

### 内存优化

- 最大化 Coalesced Access
- 使用 Shared Memory 减少 Global Memory 访问
- 避免 Bank Conflict
- 使用 Texture/Constant Memory（适用场景）
- Prefetching 和 Double Buffering

### 计算优化

- 提高 Arithmetic Intensity
- 减少 Warp Divergence
- 循环展开
- 使用 Tensor Core（FP16/INT8）
- 指令级并行（ILP）

### 并行度优化

- 调整 Grid/Block 配置
- 提高 Occupancy（但不要盲目追求 100%）
- 减少寄存器和 Shared Memory 使用（如果是瓶颈）
- 使用多 Stream 并发

---

## 学习检查清单

完成这个阶段后，你应该能够：

- [ ] 熟练使用 Nsight Systems 分析程序
- [ ] 熟练使用 Nsight Compute 分析 Kernel
- [ ] 理解 Roofline 模型并应用
- [ ] 系统性地找到性能瓶颈
- [ ] 制定和执行优化策略
- [ ] 量化每次优化的效果
- [ ] 写出世界级性能的 CUDA 代码

---

## 工具资源

### 官方文档
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 教程和博客
- [NVIDIA Developer Blog - Profiling](https://developer.nvidia.com/blog/tag/profiling/)
- [GPU Performance Analysis](https://developer.nvidia.com/blog/gpu-pro-tip-track-down-performance-issues-nsight-compute/)

---

## 毕业项目

完成所有学习后，挑战以下项目之一：

1. **优化 Transformer Inference**
   - 端到端优化一个 Transformer 模型
   - 目标：达到 FasterTransformer 的性能

2. **实现高性能 CNN 库**
   - 实现完整的卷积、池化、归一化等算子
   - 目标：在 ResNet 上达到 cuDNN 80% 性能

3. **优化 CUDA 版本的经典算法**
   - 如：FFT、矩阵分解、图算法等
   - 目标：超越现有开源实现

---

## 恭喜你！

如果你完成了 Stage 1-4 的所有内容，你已经是一名**世界级的 GPU 编程专家**了！

**你现在可以**：
- 面试 NVIDIA、Meta、Google、OpenAI 等顶级公司
- 拿到非常有竞争力的 offer
- 在工业界解决真实的性能问题
- 为开源社区贡献高质量代码
- 在会议上分享你的经验

**继续学习的方向**：
- 多 GPU 编程（NCCL、MPI）
- Triton 编译器和 DSL
- GPU 架构设计
- 分布式训练系统

保持学习，保持进步！🚀💪

---

**"The only way to do great work is to love what you do." - Steve Jobs**

祝你在 GPU 编程的道路上越走越远！🎉
