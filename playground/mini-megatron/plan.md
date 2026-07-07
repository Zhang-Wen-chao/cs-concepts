# mini-megatron 计划

> 用 **1% 的代码量**复现 Megatron-LM 核心并行策略(TP+PP+DP+AMP),验证"简洁实现是否也能高效"。

## 动机

- 学习 Megatron 的核心机制:分布式 Transformer 训练
- 用 **~3,000 行代码**挑战 **~300,000 行**的 Megatron-LM,看性能差距有多大
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

## 当前硬件环境

```
主要开发机:
  GPU: 4 × NVIDIA L20 (48 GB, Ada Lovelace, 无 NVLink)
  CPU: Intel Xeon Gold 6530 × 2 (32 核 × 2 超线程 = 128 线程)

备用机(同驱动版本):
  GPU: 4 × NVIDIA GeForce RTX 4090 D (24 GB, 无 NVLink)
  Driver: 550.127.05 / CUDA 12.4
```

L20 显存充裕(48 GB)但无 NVLink(4090D 也无 NVLink),TP 场景 all-reduce 走 PCIe。
由于当前只有 4 卡,Phase 3 暂缓,待未来 8 卡服务器就绪后启动。

## 环境准备

当前机器网络受限,无法直连 HuggingFace Hub(被墙),需通过镜像站下载 GPT-2 tokenizer:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

PyTorch TF32 默认关闭,训练前手动开启以提升约 30% 吞吐:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

现有依赖已安装(无需额外 pip install):

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.6.0+cu124 | CUDA 12.4 runtime, driver 550.127.05 兼容 |
| transformers | 5.8.1 | |
| datasets | 4.8.5 | |
| tokenizers | 0.22.2 | |
| numpy | 2.2.6 | |
| megatron-core | **0.15.0** | 0.16.1 依赖 NCCL 2.29.7(CUDA 13.x),本机不支持 |

> **NCCL**:当前容器 /dev/shm 仅 64MB,多卡 NCCL 通信受限。
> **建议**:重新创建容器时指定 `--shm-size 8g` 或 `--ipc=host`,否则多卡训练 (DP/PP) 会用 Socket 通信替代共享内存,吞吐大幅下降。
> **megatron-core 版本**:0.16.1 开始依赖 nvidia-nccl-cu13 2.29.7 → 需要 CUDA 13.x driver。当前 driver 550.127.05 止步于 CUDA 12.4,故用 0.15.0。0.15.0 包含我们需要的全部 API(GPTModel、TP/PP、MockGPTDataset),不影响对比基线。

## 四阶段路线

### Phase 1: 单卡 Transformer (基线)

```
单 GPU 上跑通 transformer 训练
验证 loss 下降
```

- 模型: decoder-only, 同 GPT 架构
- 数据: TinyStories (本地已缓存 200 万条,约 1.5 GB)
- 分词: HuggingFace GPT-2 tokenizer (通过 `HF_ENDPOINT=https://hf-mirror.com` 下载)
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

**硬件**:2 × NVIDIA L20
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

**硬件**:4 × NVIDIA L20 (当前上限)
**交付**:简历可写"实现了 Megatron TP+PP 双轴并行"

**验证脚本**:
```bash
torchrun --nproc_per_node=4 main.py --phase 2b --tp 2 --pp 2
```

### Phase 3: +DP + Mixed Precision (完整 mini) — 暂缓,待 8 卡就绪

```
目标: TP=2, PP=2, DP=2 → 8 GPU
当前: TP=2, PP=2, DP=1 → 4 GPU (若需提前验证)
```

- Data Parallel: 梯度 all-reduce
- Mixed Precision: bf16/fp16 训练 + loss scaling (fp16 时需要)
- Activation Recomputation(选做,几行代码)
- 分布式 Optimizer(选做,ZeRO-1 风格,只切 optimizer state)

**目标硬件**:8 × NVIDIA L20 (当前 4 卡,DP=1 可先跑通逻辑)
**交付**:简历可写"实现了 Megatron 3D 并行 + Mixed Precision"

**未来验证脚本**:
```bash
torchrun --nproc_per_node=8 main.py --phase 3 --tp 2 --pp 2 --dp 2
```

## 硬件需求

| Phase | 并行配置 | GPU 数 | 硬件 | 显存需求 |
|---|---|---|---|---|---|
| 1 | — | 1 | 1 × L20 | ~2GB |
| 2a | TP=2 | 2 | 2 × L20 | ~4GB×2 |
| 2b | TP=2, PP=2 | 4 | 4 × L20 ✅ **当前上限** | ~4GB×4 |
| 3 | TP=2, PP=2, DP=2 | 8 | 8 × L20 ⏳ **待扩容后** | ~4GB×8 |

## 模型配置

```python
config = {
    "num_layers": 12,         # PP=2 → 每 stage 6 层(bubble 更友好)
    "hidden_size": 512,       # 够演示,不爆显存
    "num_attention_heads": 8, # head_dim=64
    "ffn_hidden_size": 2048,  # 4× hidden_size
    "vocab_size": 50304,      # GPT-2 原始 50257,向上取整到 50304(可被 TP 整除)
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
├── data/                       # 数据加载 & 预处理
│   └── dataset.py               # TinyStories 加载 + tokenize
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
├── setup.sh                    # 环境变量设置(HF_ENDPOINT)
├── eval/                       # 评测 & 与 Megatron-Core 对比
│   └── run_megatron_baseline.py # Megatron-Core 基线训练(约 100 行)
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
| 数据 | TinyStories (TinyShakespeare 替代品,本地已缓存) | 1 张 L20 就能跑 |
| Attention | PyTorch `scaled_dot_product_attention` | 替代手写 SDPA |
| bf16 | `torch.cuda.amp` | 原生支持 |
| 通信重叠(TP) | `torch.cuda.Stream` + `torch.distributed.all_reduce` + layer compute 非阻塞 | 不需要手写 CUDA kernel |
| 通信重叠(PP) | `torch.cuda.Stream` + `batch_isend_irecv` | 不需要手写 CUDA kernel |

## 不做的事

- ❌ 手写 CUDA kernel (这是 NVIDIA 的工作,mini 版本不需要)
- ❌ 实现 Flash Attention (直接复用 PyTorch 的)
- ❌ Sequence Parallelism (第 2 版考虑)
- ❌ MoE / Expert Parallelism (超出 mini 范围)

## 验证方式

每个 Phase 完成后,从以下维度评测,逐级对比:

### 1. Loss 等价性（数学正确性）

同一批数据、相同配置下,多卡 loss 曲线必须与单卡完全重合。

```text
Phase 1:   loss 5.46 → 5.45 → 5.44 → ...  (随机数据基线)
Phase 2a:  loss 5.46 → 5.45 → 5.44 → ...  (数学等价)
```

差值 > 1e-3 说明并行实现有 bug。
> 注:使用随机数据不收敛,loss 在 ~5.46 附近波动属正常。

### 2. 吞吐量 (Throughput)

**tokens_per_step 计算规则:只有 DP(数据并行)增加 token 总数,TP/PP 拆分模型,不增加 token。**

```
tokens_per_step = micro_batch_size × seq_len × dp_world
  dp_world = total_GPUs / (tp × pp)

示例: TP=2 时 dp_world=1,2 张卡处理同一份数据
      DP=2 时 dp_world=2,各处理不同数据
```

| Phase | 配置 | tok/s | Scaling Eff | 说明 |
|---|---|---|---|---|
| 1 | 1 GPU (TP=1 PP=1 DP=1) | **25,825** | — | 基线 |
| 2a | TP=2 (2 GPU, DP=1) | **12,964** | ~25% | TP 通信开销 > 计算收益 |
| DP | DP=2 (2 GPU, TP=1 PP=1) | **24,805** | ~48% | 缺 SHM,走 Socket |
| 2b | TP=2 PP=2 (4 GPU) | TBD | TBD | 待重建容器 |
| 3 | TP=2 PP=2 DP=2 (8 GPU) | TBD | TBD | 待硬件就绪 |

> **TP=2 仅 12,964 tok/s**(单卡 25,825 的一半):512 hidden 太小,all-reduce 通信占主导。Megatron 官方建议 hidden ≥ 4096 才用 TP。
> **DP=2 仅 24,805 tok/s**(接近单卡但略低):容器 /dev/shm 仅 64MB,被迫 `NCCL_SHM_DISABLE=1`,梯度同步走 Socket。重建容器即可解决。
> Scaling Efficiency = 实际吞吐 / (GPU数 × 单卡吞吐)。

### 3. MFU (Model FLOP Utilization)

行业标准指标,衡量 GPU 算力利用率:

```
MFU = (24 × L × h² × B × s × 4.5 + 6 × L × h × s × V) × dp_world × num_steps
     ─────────────────────────────────────────────────────────────────
     elapsed × total_GPUs × GPU_peak_FLOPS

  L = num_layers, h = hidden_size, B = micro_batch_size, s = seq_len, V = vocab_size
  GPU_peak = L20 FP16 ~96 TFLOPS
```

- L20 FP16/TF32 峰值: ~96 TFLOPS
- 通过 `torch.cuda.cudart().cudaProfilerStart()` 或理论计算 flop 获取
- Megatron 官方在 H100 上报 ~47%
- 实测 Megatron-Core 基线: **Phase 1 = 41.56% MFU, Phase 2a(TP=2) = 10.47% MFU**(小模型 TP 降低算力利用率)

### 4. 每卡峰值显存

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

验证并行策略是否有效降低每卡显存。

### 5. 通信开销占比

```bash
NCCL_DEBUG=WARN torchrun ... 2>&1 | grep "all_reduce\|send_recv" | wc -l
```

通信时间占比 = 通信耗时 / 总迭代时间,衡量重叠优化的效果。

## 与 Megatron-LM 对比

### Megatron-LM 现状

NVIDIA 已将 Megatron 拆为两层:

| 组件 | 安装方式 | 用途 |
|---|---|---|
| **megatron-core** | `pip install megatron-core` | 核心库(TP/PP/DP 原语 + 模型定义) |
| **Megatron-LM** | git clone | 训练脚本 + 示例 |

当前环境 **已安装 `megatron-core==0.15.0`**（配合 torch 2.6.0+cu124,NVIDIA driver 550.127.05 完全兼容)。

```bash
pip install megatron-core==0.15.0
```

> 0.16.1 依赖 nvidia-nccl-cu13(CUDA 13.x),本机 driver 550.127.05 只支持到 CUDA 12.4,故使用 0.15.0。
> 0.15.0 已包含 GPTModel、TP/PP/DP、MockGPTDataset 等全部需要 API,不影响对比基线。
> 本机容器 /dev/shm 仅 64MB,多卡评测需加环境变量: `NCCL_SHM_DISABLE=1`。

### 对比方式

核心思路：用相同的模型配置(12层/512dim/8头/512seq)、相同的 mock 数据、相同的 TP/PP 设置,跑两个框架。

#### 评测脚本

`megatron-core` 的 example 脚本不在 pip 包内(在 GitHub repo 中,当前网络不可达)。我们会写一个轻量评测脚本 `eval/run_megatron_baseline.py` (约 100 行),核心逻辑:

```python
# Megatron-Core 基线:创建 GPTModel + 训练 N 步,记录 loss 和吞吐
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

config = TransformerConfig(num_layers=12, hidden_size=512, ...)
model = GPTModel(config=config, transformer_layer_spec=get_gpt_layer_local_spec(), ...)
# 训练循环,每步打印 loss 和 tok/s
```

mini-megatron 侧直接用 `main.py` 跑相同配置。

#### 运行命令

```bash
# Phase 1 基线 (单卡)
torchrun --nproc_per_node=1 eval/run_megatron_baseline.py --tp 1 --pp 1  # Megatron
torchrun --nproc_per_node=1 main.py --phase 1                           # mini

# Phase 2a (TP=2)
torchrun --nproc_per_node=2 eval/run_megatron_baseline.py --tp 2 --pp 1
torchrun --nproc_per_node=2 main.py --phase 2a --tp 2

# Phase 2b (TP=2, PP=2)
torchrun --nproc_per_node=4 eval/run_megatron_baseline.py --tp 2 --pp 2
torchrun --nproc_per_node=4 main.py --phase 2b --tp 2 --pp 2
```

#### 采集指标

| 维度 | 方法 | 对标 Megatron |
|---|---|---|
| **Loss 等价性** | 每 step 打印 loss,合并画图,验证曲线重合 | ✅ math equivalence |
| **吞吐量** | 总 tokens / 总时间 = tok/s,算 scaling efficiency | ✅ 对标 weak scaling |
| **MFU** | 实际 FLOP/s ÷ GPU 峰值 FLOP/s (L20: ~96 TFLOPS) | ✅ **行业标准, Megatron 报 47%** |
| **每卡峰值显存** | `torch.cuda.max_memory_allocated()` | ✅ 显存效率 |
| **通信开销占比** | `NCCL_DEBUG=WARN` 统计 comm_time / total | ✅ 验证重叠优化 |
| **代码量** | `cloc megatron/core/` vs `cloc mini-megatron/` | 1% 代码量 |

> 如果 **1% 的代码量跑出 80%+ 的性能**,这就是项目的核心价值。

### 代码量对比

| 项目 | Python 文件数 | 代码量 |
|---|---|---|
| Megatron-LM (core) | ~1,119 | ~300,000 行 |
| mini-megatron | ~15 | ~3,000 行 |
| **占比** | **1.3%** | **1%** |
| 功能覆盖 | TP+PP+DP+AMP | TP+PP+DP+AMP |

### megatron-core 版本路线(与我们的关系)

| 版本 | 新增的主要功能 | 影响我们吗？ |
|---|---|---|
| 0.1.0 ~ 0.7.0 | 基础 TP/PP/DP 原语 | ✅ 核心功能齐全 |
| 0.8.0 ~ 0.12.0 | MoE(混合专家)、DDP 优化、Context Parallel 增强 | ❌ mini 不做 MoE/CP |
| 0.13.0 ~ 0.15.0 | FP8 训练、Inference Engine、Mamba、多模态 | ❌ L20 不支持 FP8,不做推理 |
| **0.16.1 (已安装)** | 最新版,`torch>=2.6.0`,含全部功能 | ✅ **对比基线完全够用** |

> 0.8.0 → 0.16.1 之间的更新主要围绕:MoE(混合专家,路由到不同子网络)、FP8(H100 精度格式)、CP(序列并行)、Inference(推理服务)——**mini-megatron 全都不涉及**,0.16.1 作为对比基线完全够用。

### 注意事项

- Megatron-Core 0.16.1 有 TE 时用 fused kernel,无 TE 时 fallback；mini 版本全用 PyTorch native
- 对比时关掉 Megatron 的额外功能(Sequence Parallelism、分布式 optimizer 等)
- mini 版本的吞吐差距主要来自:缺少 fused kernel、无通信优化(NVLink 等)

## 仓库策略

```
开发阶段: cs-concepts/playground/mini-megatron/ 内开发
发布阶段: Phase 2a 跑通后 → 抽离为独立 GitHub 仓库
```

原因:
- 开发期间跟已有笔记(`playground/megatron-tp-dp-switch.md`)在一起,方便 review
- 独立仓库更适合简历展示,commit 历史完整
- Phase 2a 是第一个可演示的里程碑(TP 切分 + loss 等价),此时抽离时间点最佳

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