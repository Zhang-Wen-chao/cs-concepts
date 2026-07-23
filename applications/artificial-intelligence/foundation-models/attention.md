# Attention 机制

## 一句话

每个 token 问一遍所有其他 token："你对我多重要？"，然后按重要性加权融合信息。

---

## 从"查询"理解

假设你要在图书馆找一本书：

```
你 → query（你要找什么）
每本书的标题 → key（每本书是什么）
找到书后的内容 → value（实际信息）
```

Attention 就是让每个 token 扮演这三个角色。

---

## 权重在哪里

很多人第一次看到公式 `Attention(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V` 会困惑：权重在哪？

**完整的计算图（含权重）：**

```
                       X (上一层的输出)
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
       [@ W_q]       [@ W_k]       [@ W_v]    ← W_q, W_k, W_v 才是可训练的参数
          ↓             ↓             ↓
          Q             K             V
          └─────────┬──┘             │
                    ↓                │
              Q @ Kᵀ ÷ √d_k          │
                    ↓                │
               softmax              │
                    ↓                │
               [权重矩阵]            │
                    └────── @ V ─────┘
                              ↓
                            输出
```

**Q、K、V 是 X 乘以权重矩阵算出来的值**（相当于隐藏层的输出值，不是权重本身）。

```
X (上一层的输出, 形状 [seq_len, d])

Q = X @ W_q   形状 [seq_len, d_k]
K = X @ W_k   形状 [seq_len, d_k]
V = X @ W_v   形状 [seq_len, d_v]
```

W_q、W_k、W_v 跟卷积核、线性层的 W 一样——**随机初始化，反向传播更新**。

注意 Q 和 K 不是凭空产生的，也不是自己去哪里取回来的——就是 X 分别跟三个不同矩阵做乘法，投影到三个不同空间去干不同的活。

### Attention 计算

```
Attention(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V
```

三步：

```
1. Q @ Kᵀ          → 相似度矩阵 [seq_len, seq_len]
                       位置 (i,j) 表示 token i 对 token j 的关注度

2. softmax( / √d_k) → 归一化成概率（每行和为 1）

3. @ V              → 按权重融合所有 token 的值
```

### 为什么要除以 √d_k？

d_k 越大，点积的方差越大，softmax 会趋向 one-hot（仅一个位置接近 1，其余接近 0）。除 √d_k 把方差拉回 1，梯度更好。

---

## Multi-Head Attention

不做一次 attention，而是把 Q/K/V 分成 h 个头，每个头独立做，然后拼起来：

```
Q → [头1, 头2, ..., 头h]    每个头形状 [seq_len, d_k/h]
K → [头1, 头2, ..., 头h]
V → [头1, 头2, ..., 头h]

每个头: head_i = Attention(Q_i, K_i, V_i)
输出: Concat(head_1, ..., head_h) @ W_o
```

**每个头可以关注不同角度的关系**（语法、语义、距离等）。

---

## Self-Attention vs Cross-Attention（了解即可）

| | Q 来自 | K/V 来自 | 用在哪 |
|---|---|---|---|
| Self-Attention | 自己 | 自己 | GPT 的每一层（你在学的） |
| Cross-Attention | 解码器 | 编码器 | 翻译模型（Encoder-Decoder） |

**GPT 只用 Self-Attention，没有 Cross-Attention。** Cross-Attention 是翻译模型里解码器回头去看源语言用的，跟你的 Megatron 实验无关。可跳过。

---

## Masked Self-Attention

GPT 生成时不能看未来的 token。所以 attention 矩阵的上三角被 mask 掉（设成 -∞，softmax 后为 0）：

```
token 1: [0.9, 0.1,  -∞,  -∞]
token 2: [0.4, 0.4, 0.2,  -∞]
token 3: [0.3, 0.3, 0.2, 0.2]
```

---

## 交叉引用

- [Transformer：完整的架构](transformer.md)
- 代码实践：`practices/06_attention_numpy.py`（手写 Attention）
