# Transformer 架构

## 一句话

只靠 Attention + 前馈网络，不用 RNN 的循环结构，并行处理整个序列。

---

## GPT 使用的 Decoder-only 结构

```
输入 token IDs [1, 23, 45, 67, ...]
    ↓
Embedding [151936, 1024] → 每个 token 查表变成向量
    ↓  + Position Encoding（告诉模型位置信息）
    ↓
Block 1:
  ├── Masked Self-Attention（每个 token 看前面的所有 token）
  │    └── Q/K/V 投影 → attention → 拼接 → W_o 投影
  ├── + 残差连接（Add）→ LayerNorm
  ├── FFN（SwiGLU）
  │    └── gate/up/down 三层线性 + 激活
  └── + 残差连接（Add）→ LayerNorm
    ↓
Block 2 ... Block N（Qwen3 有 28 层）
    ↓
LayerNorm
    ↓
LM Head [151936, 1024] → 预测下一个 token 的概率
```

---

## 每个 Block 的内部

### Attention 子层

```
输入 X [seq_len, 1024]
    ↓
Q = X @ W_q    [1024, 1024]    → 投影
K = X @ W_k    [1024, 1024]
V = X @ W_v    [1024, 1024]
    ↓
分头（16 个头，每头 d=64）→ Masked Self-Attention → 合并
    ↓
output = Concat(heads) @ W_o  [1024, 1024]
    ↓ 残差连接
output = X + output
    ↓ LayerNorm
```

### FFN 子层（SwiGLU）

```
输入 X [seq_len, 1024]
    ↓
gate = X @ W_gate  [3072, 1024]  → SiLU 激活
up   = X @ W_up    [3072, 1024]
    ↓
hidden = gate * up               → 元素乘
    ↓
output = hidden @ W_down  [1024, 3072]
    ↓ 残差连接
output = X + output
    ↓ LayerNorm
```

---

## 为什么有残差连接？

```
output = X + Attention(X)
       = X + FFN(X)
```

不加残差的话，梯度从第 28 层传到第 1 层要穿 28 个 Attention + 28 个 FFN，链式法则连乘极易消失或爆炸。

残差连接 = **给梯度开一条直达通道**，让深层网络也能训练。

---

## 位置编码

Self-Attention 本身没有顺序概念（交换任意两个 token 结果不变）。

给每个位置加一个**位置编码向量**，告诉模型 token 的先后关系。

GPT/Qwen 用 **RoPE（旋转位置编码）**——把位置信息编码成旋转矩阵，乘到 Q 和 K 上，效果好且能外推更长序列。

---

## Qwen3-0.6B 具体参数

从上面抽象到具体：

| 参数 | 值 |
|---|---|
| hidden_size | 1024 |
| num_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads（GQA） | 8 |
| head_dim | 128 |
| ffn_hidden (intermediate) | 3072 |
| vocab_size | 151936 |
| 总参数量 | ~0.6B |

### 参数量怎么算

```
Embedding：
  W = vocab_size × hidden_size = 151936 × 1024 = 155M

每层 Attention（GQA：Q 16 个头，K/V 8 个头）：
  W_q = hidden × (heads × head_dim) = 1024 × (16 × 128) = 2.1M
  W_k = hidden × (kv_heads × head_dim) = 1024 × (8 × 128) = 1.0M
  W_v = hidden × (kv_heads × head_dim) = 1024 × (8 × 128) = 1.0M
  W_o = (heads × head_dim) × hidden = (16 × 128) × 1024 = 2.1M
  小计：6.3M

每层 FFN（SwiGLU）：
  W_gate = hidden × ffn_hidden = 1024 × 3072 = 3.1M
  W_up   = hidden × ffn_hidden = 1024 × 3072 = 3.1M
  W_down = ffn_hidden × hidden = 3072 × 1024 = 3.1M
  小计：9.4M

每层合计：6.3M + 9.4M = 15.7M

总参数：
  Embedding：          155M
  28 层：  28 × 15.7M = 440M
  LayerNorm + bias：   ~5M
  ────────────────────────
  合计：              ~600M = 0.6B
```

权重大头在 Embedding（155M）和 28 层 FFN（263M）。Attention 虽名字响亮，只占 ~176M。

---

## 交叉引用

- [Attention 机制详解](attention.md)
- [Megatron 分布式训练：怎么把 Transformer 拆到多卡](../../training-inference/01-distributed.md)
- [实验验证：TP/DP/PP Reshard 实践](https://github.com/Zhang-Wen-chao/megatron-tp-reshard)
