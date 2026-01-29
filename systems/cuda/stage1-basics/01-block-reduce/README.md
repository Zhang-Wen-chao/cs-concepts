# 01. Block Reduce - 块内规约

## 学习目标

- 理解并行规约（Parallel Reduction）的基本思想
- 掌握 `__syncthreads()` 的使用
- 对比不同规约策略的性能差异
- 理解 warp divergence 的影响

## 什么是 Block Reduce？

Block Reduce 是在一个线程块内对所有元素进行规约操作（如求和）。

例如，将 8 个数字求和：
```
输入: [1, 2, 3, 4, 5, 6, 7, 8]
输出: 36
```

## 两种实现策略

### 1. Neighbored Pairing（邻居配对）

相邻元素配对相加：

```
步骤1 (stride=1): [1+2, 3+4, 5+6, 7+8] = [3, 7, 11, 15]
步骤2 (stride=2): [3+7, 11+15]         = [10, 26]
步骤3 (stride=4): [10+26]              = [36]
```

**问题**：会导致 warp divergence（线程分化），性能较差。

### 2. Interleaved Pairing（交错配对）

间隔元素配对相加：

```
步骤1 (stride=4): [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
步骤2 (stride=2): [6+10, 8+12]         = [16, 20]
步骤3 (stride=1): [16+20]              = [36]
```

**优势**：活跃线程连续，减少 warp divergence，性能更好。

## 任务要求

1. 实现 `reduceNeighbored()` 函数
2. 实现 `reduceInterleaved()` 函数
3. 对比两种方法的性能

## 编译运行

```bash
# 编译
make

# 运行（默认 block size = 256）
./block_reduce

# 指定 block size
./block_reduce 512
```

## 预期输出

```
Device 0: NVIDIA GeForce RTX 3090
  Compute Capability: 8.6
  ...

Array size: 16777216
Grid size: 65536, Block size: 256

CPU reduce:           XXXXXX, elapsed X.XXXXXX sec
GPU Neighbored:       XXXXXX, elapsed X.XXXXXX ms ✓
GPU Interleaved:      XXXXXX, elapsed X.XXXXXX ms ✓
```

**Interleaved 应该比 Neighbored 快！**

## 实现提示

### reduceNeighbored

```cuda
for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
        idata[tid] += idata[tid + stride];
    }
    __syncthreads();
}
```

### reduceInterleaved

```cuda
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        idata[tid] += idata[tid + stride];
    }
    __syncthreads();
}
```

## 关键概念

### 1. `__syncthreads()`

**为什么需要？**
- 确保所有线程完成当前步骤后，再进行下一步
- 避免数据竞争（race condition）

**何时使用？**
- 在 shared memory 或 global memory 上有数据依赖时
- 一个线程写，另一个线程读

### 2. Warp Divergence

**什么是 Warp？**
- GPU 以 32 个线程为一组（warp）调度执行
- 同一个 warp 中的线程应该执行相同的指令

**Divergence 的影响：**
- `Neighbored` 方法：随着 stride 增大，越来越多线程闲置
- `Interleaved` 方法：活跃线程始终连续，减少 divergence

### 3. Block-level vs Device-level Reduce

**当前实现**：Block-level
- 每个 block 输出一个结果
- 需要在 CPU 上对这些结果再次求和

**完整的规约**：Device-level
- 在 GPU 上递归调用 kernel
- 或使用 CUB 库的 `DeviceReduce`

## 思考题

1. 为什么 Interleaved 比 Neighbored 快？
2. 如果不使用 `__syncthreads()` 会发生什么？
3. 如何在 GPU 上完成完全的规约（不需要 CPU 参与）？
4. 最后几个 warp 还需要 `__syncthreads()` 吗？

## 扩展学习

完成基础实现后，可以尝试：

1. **优化最后一个 warp**
   - 同一个 warp 内线程自动同步
   - 可以去掉最后几步的 `__syncthreads()`

2. **使用 Shared Memory**
   - 当前实现直接在 global memory 上操作
   - 下一节会学习如何使用 shared memory 优化

3. **完全展开循环**
   - 如果 block size 固定，可以完全展开循环
   - 提高性能

## 参考资料

- [CUDA Samples - Reduction](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/reduction)
- [Mark Harris - Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

---

完成后打勾：
- [ ] 实现 reduceNeighbored
- [ ] 实现 reduceInterleaved
- [ ] 验证结果正确
- [ ] 对比性能差异
- [ ] 理解 warp divergence 的影响
- [ ] 写学习笔记

下一步：[02-matrix-transpose](../02-matrix-transpose/)
