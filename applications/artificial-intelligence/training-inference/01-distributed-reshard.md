# 分布式训练：Checkpoint Reshard 原理

## 一句话

不同并行配置（TP/DP/PP）间恢复训练 = 用 `ShardedTensor` 描述"我存了什么"和"我要什么"，框架自动做 slice-level 的字节重新分配。

---

## 问题

TP=2 训练了 100 步，保存了 checkpoint。想加载到 TP=4 继续训练，怎么办？

```
TP=2 保存：
  rank 0: q_proj 的 [0:512] 列     ← 每个 rank 存了矩阵的一半
  rank 1: q_proj 的 [512:1024] 列

TP=4 加载：
  rank 0: 想要 q_proj 的 [0:256] 列  ← 每个 rank 只要四分之一
  rank 1: 想要 q_proj 的 [256:512] 列
  rank 2: 想要 q_proj 的 [512:768] 列
  rank 3: 想要 q_proj 的 [768:1024] 列
```

---

## 解法：ShardedTensor（元数据驱动）

### 保存时

每个 rank 把 local tensor 包成 `ShardedTensor`，编码它在全局张量中的坐标：

```python
ShardedTensor.from_rank_offsets(
    key="q_proj.weight",
    data=W[:, :512],        # 本地 tensor [1024, 512]
    (1, 0, 2),             # (axis=1, offset=0, total=2)
)
```

内部存了：
- `global_shape: [1024, 1024]`
- `global_offset: [0, 0]`
- `local_shape: [1024, 512]`
- `axis_fragmentations: [1, 2]` → 第 1 维切成 2 份，我持有第 0 份

### 加载时

新拓扑（TP=4）启动，每个 rank 按**新的**切分方式构造 `ShardedTensor`：

```python
ShardedTensor.from_rank_offsets(
    key="q_proj.weight",
    data=W[:, :256],        # 新 local shape
    (1, 0, 4),             # axis=1 切成 4 份，我要第 0 份
)
```

### 自动匹配

PyTorch Distributed Checkpoint (DCP) 对比两套元信息：

```
已存 2 个 shard: [0:512], [512:1024]
请求 4 个 shard: [0:256], [256:512], [512:768], [768:1024]
```

DCP 发现每个新 shard 都在某个已存 shard 的连续区间内，**直接按 offset + length 从文件读取**。

```
rank 0 ← 读 saved_shard_0 的 [0:256]
rank 1 ← 读 saved_shard_0 的 [256:512]
rank 2 ← 读 saved_shard_1 的 [0:256]
rank 3 ← 读 saved_shard_1 的 [256:512]
```

**无需跨 rank 通信，无需合并再拆分。**

---

## 更复杂的情况

### PP 变化

不同 PP stage 保存不同层的参数。key 包含层索引（如 `model.layers.0.self_attn.q_proj.weight`），DCP 按 key 精确匹配，自动路由到正确的文件。

### DP 变化

Optimizer state（`exp_avg`、`exp_avg_sq`）的形状跟模型参数一一对应，也被包成 `ShardedTensor`。DP 变化意味着 optimizer shard 需要拆分或合并，同样由 DCP 自动做 slice-level 映射。

---

## 三种并行的 Load/Reshard 对比

| 并行 | Reshard 时做什么 | 通信 |
|---|---|---|
| TP 变化 | tensor 的列/行重新切片 | 无（只读文件） |
| PP 变化 | 不同 key 的路由 | 无 |
| DP 变化 | optimizer state 重新切片 | 无 |

所有 tensor 级别的搬运都是**文件读 + offset 计算**，无需跨卡通信。

---

## 代码路径

| 模块 | 作用 |
|---|---|
| Megatron `ShardedTensor` | 编码 global shape、offset、fragmentation |
| `torch.distributed.checkpoint.load()` | 对比 saved vs requested metadata，自动 read |
| `run_phase.sh` | 传 `--load` + `--ckpt-format torch_dist` + TP/PP 参数 |

---

## 和实验的关系

[megatron-tp-reshard](https://github.com/Zhang-Wen-chao/megatron-tp-reshard) 验证的就是这个机制在 6 种拓扑间的正确性：

```
P0: tp2dp2pp1 → tp4dp1pp1  ✅ mean diff < 6e-3
P1: 所有 TP/DP/PP 组合      ✅ 全部通过
P2: 纯 DP ↔ 纯 PP          ✅ 全部通过
```

---

## 交叉引用

- [TP/DP/PP 基础概念](README.md)
- [Transformer 架构详解](../foundation-models/transformer.md)
- [Megatron TP/DP 切换实验](../../playground/megatron-tp-dp-switch.md)
