# mini-megatron 计划

> 实现一个极简版的 Megatron-LM,覆盖 Tensor Parallelism + Pipeline Parallelism + Data Parallelism + Mixed Precision 四大分布式训练核心功能。

## 动机

- 学习 Megatron 的核心机制:分布式 Transformer 训练
- 做出可演示的 GitHub 项目,作为 AI Infra 技能证明
- 把 cs-concepts 的 C++ / CUDA / 分布式知识串联起来

## 范围

实现 Megatron-LM 的核心功能(按优先级):

| # | 功能 | Megatron-LM | mini-megatron |
|---|---|---|---|
| 1 | **Tensor Parallelism** | Column/Row Linear split + all-reduce | **做** |
| 2 | **Pipeline Parallelism** | 1F1B 调度,micro-batch,点对点通信 | **做** |
| 3 | **Data Parallelism** | 梯度同步 all-reduce | **做** |
| 4 | **Mixed Precision** | bf16/fp16 + loss scaling (fp16 时需要) | **做** |
| 5 | **梯度累积 (Gradient Accumulation)** | PP micro-batch 配合,控制实际 batch size | **做** |
| 6 | **通信-计算重叠 (Comm-Compute Overlap)** | CUDA stream + batch_isend_irecv 重叠 all-reduce | **做(简单版)** |
| 7 | Sequence Parallelism | 沿序列维度切分 activation | **第 2 版** |
| 8 | Activation Recomputation | 用计算换显存 | **Phase 3 选做** |
| 9 | Checkpoint | 保存/加载权重 + 可重回 | **做(简单版)** |
| 10 | 分布式 Optimizer | ZeRO-1 风格 (Megatron 自带 `distrib_optimizer.py`) | **Phase 3 选做** |

### 核心功能说明

**梯度累积**:PP 的 1F1B 调度默认配合 gradient accumulation。一个 optimizer step 内跑 N 个 micro-batch,梯度累加后才更新一次。关键参数:
- `grad_accum_steps` 控制通信频率
- 实际 batch size = micro_batch × DP × grad_accum
- Loss 需按累加步数缩放

**通信-计算重叠**:不是优化,是 Megatron 的核心特征。TP 的 all-reduce 不跟下一层计算重叠 = 浪费 100% 通信时间。mini 版本用 `torch.cuda.Stream` + `batch_isend_irecv` 做简单重叠即可,不需要手写 CUDA kernel。

## 四阶段路线

### Phase 1: 单卡 Transformer (基线)

```
单 GPU 上跑通 transformer 训练
验证 loss 下降
```

- 模型: decoder-only, 同 GPT 架构
- 数据: TinyShakespeare
- 分词: HuggingFace GPT-2 tokenizer
- 框架: Python + PyTorch

**目标**:loss 从 ~8 降到 ~4,确认训练 loop 正确。

**验证脚本**:
```bash
torchrun --nproc_per_node=1 main.py --phase 1
```

### Phase 2a: Tensor Parallelism only

```
TP=2 → 2 GPU
```

- Attention 和 MLP 的 Linear 层按列/行切分
- 正确管理 f/f_b 通信模式
- 加通信-计算重叠(简单版:CUDA stream)
- 验证:跟 Phase 1 单卡 loss 曲线一致(数学等价)

**硬件**:2 × 4090
**交付**:GitHub 可发,演示 TP 切分

**验证脚本**:
```bash
torchrun --nproc_per_node=2 main.py --phase 2a --tp 2
```

### Phase 2b: + Pipeline Parallelism

```
TP=2, PP=2 → 4 GPU
```

- 按层分 stage(12 层 → 每 stage 6 层)
- 实现 1F1B 调度,micro-batch 流水线
- 梯度累积:一个 step 内跑 N 个 micro-batch
- 验证:跟 Phase 2a loss 曲线一致

**硬件**:4 × 4090
**交付**:简历可写"实现了 Megatron TP+PP 双轴并行"

**验证脚本**:
```bash
torchrun --nproc_per_node=4 main.py --phase 2b --tp 2 --pp 2
```

### Phase 3: +DP + Mixed Precision (完整 mini)

```
TP=2, PP=2, DP=2 → 8 GPU
```

- Data Parallel: 梯度 all-reduce
- Mixed Precision: bf16/fp16 训练 + loss scaling (fp16 时需要)
- Activation Recomputation(选做,几行代码)
- 分布式 Optimizer(选做,ZeRO-1 风格,只切 optimizer state)

**硬件**:8 × 4090

**验证脚本**:
```bash
torchrun --nproc_per_node=8 main.py --phase 3 --tp 2 --pp 2 --dp 2
```

## 硬件需求

| Phase | 并行配置 | 实际 GPU 数 | 最低 GPU | 显存需求 |
|---|---|---|---|---|
| 1 | — | 1 | 1 × 3090/4090 | ~2GB |
| 2a | TP=2 | 2 | 2 × 4090 | ~4GB×2 |
| 2b | TP=2, PP=2 | 4 | 4 × 4090 | ~4GB×4 |
| 3 | TP=2, PP=2, DP=2 | 8 | 8 × 4090 | ~4GB×8 |

无本地 GPU 时:AutoDL / Lambda Labs 租用,$1-2/h·卡。

## 模型配置

```python
config = {
    "num_layers": 12,         # PP=2 → 每 stage 6 层(bubble 更友好)
    "hidden_size": 512,       # 够演示,不爆显存
    "num_attention_heads": 8, # head_dim=64
    "ffn_hidden_size": 2048,  # 4× hidden_size
    "vocab_size": 50257,      # GPT-2
    "max_seq_len": 512,
}
```

约 60M 参数(embedding + LM head 共享)。

### 跟生产 Megatron 对比

| Config | GPT-3 1.3B | mini-megatron |
|---|---|---|
| num_layers | 24 | 12 |
| hidden_size | 2048 | 512 |
| num_attention_heads | 24 | 8 |
| head_dim | 85 | 64 |
| max_seq_len | 2048 | 512 |
| parameters | 1.3B | 60M |
| TP | 2 | 2 |
| PP | 2 | 2 |
| per-GPU memory | ~30GB | ~2GB |

## 代码结构

```
mini-megatron/
├── comm/                       # 通信原语(单独 review/debug)
│   ├── all_reduce.py            # TP 用的 all-reduce 封装
│   ├── send_recv.py            # PP 用的 p2p 通信
│   ├── overlap_tp.py           # TP: all-reduce ↔ layer compute 重叠
│   └── overlap_pp.py           # PP: batch_isend_irecv ↔ micro-batch 重叠
├── model/
│   ├── transformer.py          # Decoder layer, attention, MLP
│   ├── embedding.py
│   └── loss.py
├── parallel/
│   ├── tensor_parallel.py      # Linear splits + all-reduce
│   ├── pipeline_parallel.py    # 1F1B + micro-batch + gradient accumulation
│   └── data_parallel.py        # Gradient sync
├── trainer.py                  # Training loop
├── config.py
├── checkpoint.py
├── main.py                     # Entry point + arg parsing
└── scripts/
    ├── run_phase1.sh           # 单卡基线
    ├── run_phase2a.sh          # TP=2
    ├── run_phase2b.sh          # TP=2 PP=2
    └── run_phase3.sh           # TP=2 PP=2 DP=2
```

## 技术选型

| 层 | 选择 | 理由 |
|---|---|---|
| 框架 | PyTorch | 灵活,社区大 |
| 分布式通信 | `torch.distributed` + NCCL | 生产级,不需要自己写通信 |
| 数据 | TinyShakespeare 或几 MB 文本 | 1 张 4090 就能跑 |
| Attention | PyTorch `scaled_dot_product_attention` | 替代手写 SDPA |
| bf16 | `torch.cuda.amp` | 原生支持 |
| 通信重叠(TP) | `torch.cuda.Stream` + `_all_reduce` + layer compute 非阻塞 | 不需要手写 CUDA kernel |
| 通信重叠(PP) | `torch.cuda.Stream` + `batch_isend_irecv` | 不需要手写 CUDA kernel |

## 不做的事

- ❌ 手写 CUDA kernel (这是 NVIDIA 的工作,mini 版本不需要)
- ❌ 实现 Flash Attention (直接复用 PyTorch 的)
- ❌ Sequence Parallelism (第 2 版考虑)
- ❌ MoE / Expert Parallelism (超出 mini 范围)

## 验证方式

每个 Phase 完成后,跑同一个配置,跟前一 Phase 对比:

### Loss 等价性

```text
单卡 loss:      4.21 → 3.15 → 2.88 → ...
TP=2 loss:      4.21 → 3.15 → 2.88 → ...  (数学等价)
TP=2 PP=2 loss: 4.21 → 3.15 → 2.88 → ...  (数学等价)
```

通信开销应增加 5-15%,但计算结果**完全一致**。不一致就是实现有 bug。

### 吞吐量对比

```text
Phase 1:    1024 tok/s    (1 GPU, 基线)
Phase 2a:    850 tok/s    (2 GPU, TP 通信开销 ~17%)
Phase 2b:   3200 tok/s   (4 GPU, 接近理想 4× scaling)
Phase 3:    6000 tok/s   (8 GPU, ~5.8× ideal scaling)
```

展示"加了通信开销后 scaling 效率" — 这是 Megatron 的真正卖点(不只是"能跑",是"高效地跑")。

## 后续方向

### 第 2 版
- Sequence Parallelism: 沿序列维度切分
- 支持更大模型(7B+)验证
- 更复杂的 1F1B 调度(如 interleaved PP)

### 知识拓展
- Megatron 的 fp8 训练(NVIDIA H100+)
- DeepSpeed ZeRO-2/3
- NCCL 底层通信调优

## 参考

- Megatron-LM: <https://github.com/NVIDIA/Megatron-LM>
- 已有笔记: `playground/megatron-tp-dp-switch.md`
- 本笔记: `playground/mini-megatron/plan.md`