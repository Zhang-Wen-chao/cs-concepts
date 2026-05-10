# 模型训练与推理优化 - Training & Inference Optimization

> 大模型训练和推理的并行策略与性能优化

## 🎯 背景

单个 GPU 放不下大模型（7B、13B、70B、130B+），需要多卡甚至多机协作。这引出了两大问题：

- **训练**：如何在多卡间切分计算和显存？
- **推理**：如何让大模型响应更快、吞吐更高？

---

## 📚 训练并行策略

### 1. 数据并行 (Data Parallel, DP)

每卡一份完整模型，分 batch 训练，梯度同步。

```
Batch 0-511     Batch 512-1023    Batch 1024-1535
    GPU 0            GPU 1             GPU 2
  [完整模型]       [完整模型]         [完整模型]
      │                │                 │
      └────────────────┼─────────────────┘
                 all-reduce 梯度
                       │
                   更新参数
```

| 方面 | 说明 |
|------|------|
| 通信 | 每步一次梯度 all-reduce |
| 优点 | 简单，增加 batch size |
| 瓶颈 | 模型太大放不单卡时无法用 |

### 2. 张量并行 (Tensor Parallel, TP)

将单个算子（如 Linear、Attention）切分到多卡，每卡只算一部分。

```
GPU 0                        GPU 1
[head 0, head 1]             [head 2, head 3]
    │                            │
    └──────── all-reduce ────────┘
              输出
```

| 方面 | 说明 |
|------|------|
| 通信 | 每层都需要 all-reduce（高频） |
| 优点 | 解决单卡放不下的问题 |
| 瓶颈 | 通信量大，需要高速互联（NVLink） |

### 3. 流水线并行 (Pipeline Parallel, PP)

将模型按层切分到多卡，前一层算完传给后一层。

```
GPU 0: layers 1-8    GPU 1: layers 9-16    GPU 2: layers 17-24
  microbatch 0 ──→    microbatch 0 ──→       microbatch 0
  microbatch 1 ──→    microbatch 1 ──→       microbatch 1
  microbatch 2 ──→    microbatch 2 ──→       microbatch 2
```

| 方面 | 说明 |
|------|------|
| 通信 | 卡间传输激活值 |
| 优点 | 减少通信量，适合跨机 |
| 瓶颈 | 流水线气泡（bubble）导致 GPU 空闲 |

**Gradient Checkpointing**：存部分激活值、反向传播时重新计算，以时间换空间。

### 4. 混合并行 (3D Parallelism)

同时使用 TP + PP + DP，Megatron-LM 的标准方案：

```
数据并行组 0            数据并行组 1
  │                      │
  ├── TP=4, PP=2         ├── TP=4, PP=2
  │  GPU 0-7             │  GPU 8-15
  │                      │
  └──── all-reduce ──────┘
```

### 5. ZeRO (零冗余优化, DeepSpeed)

| 级别 | 切分内容 | 显存节省 |
|------|---------|---------|
| **ZeRO-1** | 切分 optimizer states | ~4x |
| **ZeRO-2** | 切分 optimizer + gradients | ~8x |
| **ZeRO-3** | 切分 optimizer + gradients + parameters | ~内存与GPU数成反比 |

**ZeRO vs TP**：ZeRO 不改变计算（无额外通信），TP 改变计算（每层有 all-reduce）。

---

## 📚 推理优化

### 1. KV Cache

Transformer 推理时缓存历史 token 的 Key/Value，避免重复计算。

```
Prompt: "今天我"
  Token 1 "今天" → 计算 K1,V1 → 缓存
  Token 2 "我"   → 计算 K2,V2 → 缓存
  Token 3 "想"   → 复用K1,V1,K2,V2 + 新K3,V3
```

| 方面 | 说明 |
|------|------|
| 显存消耗 | 随序列长度线性增长 |
| 优化 | PagedAttention（vLLM）用分页管理 KV Cache |

### 2. 量化 (Quantization)

将模型权重从 FP16/FP32 压缩到 INT8/INT4，减少显存占用和计算量。

| 精度 | 每参数大小 | 70B模型显存 | 相对精度 |
|------|-----------|------------|---------|
| FP16 | 2 bytes | 140 GB | 100% |
| INT8 | 1 byte | 70 GB | ~99% |
| INT4 | 0.5 byte | 35 GB | ~97% |

| 方法 | 做法 | 特点 |
|------|------|------|
| **PTQ** | 训练后量化 | 简单，精度略有损失 |
| **QAT** | 量化感知训练 | 精度更好，需要训练 |
| **GPTQ** | 逐层量化 | 基于 Hessian 矩阵，效果好 |
| **AWQ** | 逐通道量化 | 保护重要通道，精度更高 |

### 3. 推理框架对比

| 框架 | 核心能力 | 适用场景 |
|------|---------|---------|
| **vLLM** | PagedAttention + continuous batching | 在线推理服务 |
| **TensorRT-LLM** | 图优化 + 算子融合 + 量化 | 极致性能，部署场景 |
| **TGI (HF)** | 与 HuggingFace 生态集成 | 快速部署原型 |
| **llama.cpp** | 纯 CPU/GPGPU 推理 | 本地、边缘设备 |

---

## 🔗 关键权衡

```
训练场景：
  TP 多 → 通信重 → 需要高速互联（NVLink）
  PP 多 → 气泡大 → 需要大 batch 填充
  DP 多 → 梯度同步 → 带宽够用即可
  ZeRO → 比 TP 节省显存但延迟更高

推理场景：
  KV Cache 占用大 → vLLM 分页管理
  计算密集 → TensorRT 算子融合
  显存不足 → INT4 量化
```

---

## 🔗 相关概念

- [分布式系统](../../systems/distributed-systems.md) — 分布式训练的系统基础
- [Megatron TP/DP 切换实验](../../../playground/megatron-tp-dp-switch.md) — 具体实验设计

---

> 代码实现详见 [practices/training-inference/](../../../practices/training-inference/)
