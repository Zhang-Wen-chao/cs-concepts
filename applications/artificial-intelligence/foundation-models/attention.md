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

## Scaled Dot-Product Attention

### 输入

Q、K、V 三个矩阵，每行是一个 token 的向量：

```
Q: [seq_len, d_k]    → 每个 token 的查询
K: [seq_len, d_k]    → 每个 token 的键
V: [seq_len, d_v]    → 每个 token 的值
```

### 计算

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

## Self-Attention vs Cross-Attention

| | Q 来自 | K/V 来自 | 用在哪 |
|---|---|---|---|
| Self-Attention | 自己 | 自己 | Transformer 的每一层 |
| Cross-Attention | 解码器 | 编码器 | Encoder-Decoder 结构 |

GPT 只用 Self-Attention。

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
