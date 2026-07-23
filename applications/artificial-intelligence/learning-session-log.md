# 学习会话记录

> 记录与 AI 对话的学习主线，防止偏离方向。

---

## 当前目标

理解 Megatron TP/DP/PP reshard 的完整原理，并能给别人讲清楚。

---

## 学习路径（从根到叶）

```
机器学习基础
  ├── 什么是 ML（监督/无监督、模型、损失、梯度）
  ├── 线性模型（线性回归 vs 逻辑回归）
  └── 训练（梯度下降、优化器、正则化）
        ↓
深度学习
  ├── 神经网络（感知机、激活函数、隐藏层）
  ├── 反向传播（链式法则、梯度计算） ← 当前在学
  └── 主流架构（CNN/RNN/Transformer 对比）
        ↓
Transformer
  ├── Attention（QKV、Scaled Dot-Product）
  └── Decoder-only 结构（残差连接、RoPE）
        ↓
分布式训练
  ├── TP/DP/PP 原理
  └── Checkpoint Reshard（ShardedTensor 机制）
        ↓
Megatron 实验（实际在做的事）
  └── megatron-tp-reshard 仓库
```

---

## 关键词索引

| 概念 | 理解状态 | 对应文档 |
|---|---|---|
| 模型 = 带参数函数 | ✅ | 01-what-is-ml |
| 损失函数（MSE、交叉熵） | ✅ | 02-linear-models |
| 梯度下降 | ✅ | 03-training |
| 神经网络（隐藏层、激活函数） | ✅ | deep-learning/01-neural-networks |
| 反向传播（链式法则） | ✅ | deep-learning/02-backprop |
| CNN/RNN/Transformer 对比 | ✅ | deep-learning/03-architectures |
| Attention（QKV、加权） | ✅ | foundation-models/attention |
| Transformer 完整架构（残差、RoPE、FFN） | ✅ | foundation-models/transformer |
| TP/DP/PP | ❌ 下一个 | training-inference/ |
| ShardedTensor | ❌ 下一个 | training-inference/01-distributed-reshard |

---

## 未来关注

- **股票/时序预测用 Transformer** — Informer、Autoformer、PatchTST、LLM 微调
- **多模态统一架构** — 文本/图像/音频全部拆成 token，进同一套 Transformer

---

## 注意事项

- 先学透当前层，再往下走
- 保证听懂再学新的
- 文档在 cs-concepts 仓库中，随时回头翻
