"""
Attention 机制 - NumPy手写实现
理解 Transformer 的核心组件

作者: Zhang Wenchao
日期: 2025-11-20
"""

import numpy as np
import matplotlib.pyplot as plt


class ScaledDotProductAttention:
    """缩放点积注意力（Scaled Dot-Product Attention）

    核心公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    参数：
        Q (Query): 查询矩阵，表示"我在找什么"
        K (Key): 键矩阵，表示"我有什么信息"
        V (Value): 值矩阵，表示"信息的具体内容"
    """

    def __init__(self):
        self.attention_weights = None  # 保存注意力权重用于可视化

    def softmax(self, x):
        """数值稳定的 softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        参数:
            Q: (batch_size, seq_len, d_k) - 查询
            K: (batch_size, seq_len, d_k) - 键
            V: (batch_size, seq_len, d_v) - 值
            mask: (batch_size, seq_len, seq_len) - 掩码（可选）

        返回:
            output: (batch_size, seq_len, d_v)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        d_k = Q.shape[-1]

        # 1. 计算注意力分数：Q * K^T
        # (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
        scores = np.matmul(Q, K.transpose(0, 2, 1))

        # 2. 缩放（防止点积过大导致 softmax 饱和）
        scores = scores / np.sqrt(d_k)

        # 3. 应用掩码（可选，用于遮挡未来信息或padding）
        if mask is not None:
            scores = scores + (mask * -1e9)

        # 4. Softmax 归一化得到注意力权重
        attention_weights = self.softmax(scores)
        self.attention_weights = attention_weights  # 保存用于可视化

        # 5. 加权求和：Attention * V
        output = np.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention:
    """多头注意力（Multi-Head Attention）

    核心思想：并行运行多个注意力头，从不同子空间捕获信息

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W^O
    其中 head_i = Attention(Q*W^Q_i, K*W^K_i, V*W^V_i)
    """

    def __init__(self, d_model, num_heads):
        """
        参数:
            d_model: 模型维度（如512）
            num_heads: 注意力头数（如8）
        """
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 初始化权重矩阵（Xavier初始化）
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x):
        """
        将最后一维拆分成 (num_heads, d_k)

        输入: (batch_size, seq_len, d_model)
        输出: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        # 重塑为 (batch, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # 转置为 (batch, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        合并多个头的输出

        输入: (batch_size, num_heads, seq_len, d_k)
        输出: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        # 转置回 (batch, seq_len, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3)
        # 合并为 (batch, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        参数:
            Q, K, V: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)

        返回:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = Q.shape[0]

        # 1. 线性投影到 Q, K, V
        Q = np.matmul(Q, self.W_Q)  # (batch, seq_len, d_model)
        K = np.matmul(K, self.W_K)
        V = np.matmul(V, self.W_V)

        # 2. 拆分成多个头
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 对每个头计算注意力
        # 为了使用 ScaledDotProductAttention，需要重塑为 (batch*num_heads, seq_len, d_k)
        Q_reshaped = Q.reshape(-1, Q.shape[2], Q.shape[3])
        K_reshaped = K.reshape(-1, K.shape[2], K.shape[3])
        V_reshaped = V.reshape(-1, V.shape[2], V.shape[3])

        attention_output, _ = self.attention.forward(Q_reshaped, K_reshaped, V_reshaped, mask)

        # 重塑回 (batch, num_heads, seq_len, d_k)
        attention_output = attention_output.reshape(batch_size, self.num_heads, -1, self.d_k)

        # 4. 合并多个头
        output = self.combine_heads(attention_output)  # (batch, seq_len, d_model)

        # 5. 最终的线性投影
        output = np.matmul(output, self.W_O)

        return output


class SelfAttentionExample:
    """自注意力示例：理解句子中词与词的关系"""

    def __init__(self):
        pass

    def visualize_attention(self, sentence, attention_weights):
        """可视化注意力权重矩阵"""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')

        # 设置坐标轴标签
        ax.set_xticks(range(len(sentence)))
        ax.set_yticks(range(len(sentence)))
        ax.set_xticklabels(sentence, rotation=45)
        ax.set_yticklabels(sentence)

        # 添加颜色条
        plt.colorbar(im, ax=ax)

        # 在每个格子上显示数值
        for i in range(len(sentence)):
            for j in range(len(sentence)):
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", color="w", fontsize=8)

        ax.set_title('Self-Attention Weight Matrix\n(Each row shows attention distribution from query to keys)')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

        plt.tight_layout()
        plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
        print("注意力可视化已保存到 attention_weights.png")
        plt.close()


def demo_scaled_dot_product_attention():
    """演示：缩放点积注意力"""
    print("=" * 60)
    print("演示 1: 缩放点积注意力 (Scaled Dot-Product Attention)")
    print("=" * 60)

    # 创建简单示例
    batch_size = 1
    seq_len = 4  # 句子长度
    d_k = 8      # 特征维度

    # 随机初始化 Q, K, V
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    # 计算注意力
    attention = ScaledDotProductAttention()
    output, attention_weights = attention.forward(Q, K, V)

    print(f"\n输入维度:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")

    print(f"\n注意力权重矩阵 (每一行和为1):")
    print(attention_weights[0])
    print(f"\n每行和: {attention_weights[0].sum(axis=1)}")

    print(f"\n输出维度: {output.shape}")
    print(f"输出值（前2个token）:")
    print(output[0, :2])


def demo_multi_head_attention():
    """演示：多头注意力"""
    print("\n" + "=" * 60)
    print("演示 2: 多头注意力 (Multi-Head Attention)")
    print("=" * 60)

    # 参数设置
    batch_size = 2
    seq_len = 5
    d_model = 64    # Transformer 标准配置
    num_heads = 8

    # 创建输入
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    # 多头注意力
    mha = MultiHeadAttention(d_model, num_heads)
    output = mha.forward(Q, K, V)

    print(f"\n配置:")
    print(f"  模型维度 d_model: {d_model}")
    print(f"  注意力头数 num_heads: {num_heads}")
    print(f"  每个头维度 d_k: {mha.d_k}")

    print(f"\n输入维度: {Q.shape}")
    print(f"输出维度: {output.shape}")
    print(f"\n输出统计:")
    print(f"  均值: {output.mean():.4f}")
    print(f"  标准差: {output.std():.4f}")


def demo_self_attention_with_meaning():
    """演示：有实际意义的自注意力（简化的词嵌入）"""
    print("\n" + "=" * 60)
    print("演示 3: 自注意力可视化（句子理解）")
    print("=" * 60)

    # 简单的句子和词嵌入
    sentence = ["我", "爱", "深度", "学习"]

    # 手动设计的词向量（实际中应该用训练好的embedding）
    # 这里简化：每个词用一个4维向量表示
    word_embeddings = np.array([
        [1.0, 0.0, 0.0, 0.5],  # 我（主语特征）
        [0.0, 1.0, 0.5, 0.0],  # 爱（动词特征）
        [0.0, 0.0, 1.0, 0.5],  # 深度（形容词）
        [0.0, 0.5, 1.0, 0.5],  # 学习（名词）
    ])

    # 增加 batch 维度
    embeddings = word_embeddings[np.newaxis, :, :]  # (1, 4, 4)

    # 自注意力（Q=K=V，即输入的词向量）
    attention = ScaledDotProductAttention()
    output, attention_weights = attention.forward(embeddings, embeddings, embeddings)

    print(f"\n句子: {' '.join(sentence)}")
    print(f"\n注意力权重矩阵:")
    print(attention_weights[0])

    print(f"\n解读:")
    for i, word in enumerate(sentence):
        weights = attention_weights[0, i]
        max_idx = weights.argmax()
        print(f"  '{word}' 最关注 '{sentence[max_idx]}' (权重: {weights[max_idx]:.3f})")

    # 可视化
    visualizer = SelfAttentionExample()
    visualizer.visualize_attention(sentence, attention_weights[0])


def demo_masked_attention():
    """演示：掩码注意力（用于解码器，不能看到未来信息）"""
    print("\n" + "=" * 60)
    print("演示 4: 掩码注意力 (Masked Attention)")
    print("=" * 60)

    seq_len = 4
    d_k = 8

    # 创建输入
    np.random.seed(42)
    Q = np.random.randn(1, seq_len, d_k)
    K = np.random.randn(1, seq_len, d_k)
    V = np.random.randn(1, seq_len, d_k)

    # 创建下三角掩码（防止看到未来）
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask[np.newaxis, :, :]  # 增加 batch 维度

    print("\n掩码矩阵（1表示被遮挡）:")
    print(mask[0])

    # 无掩码注意力
    attention = ScaledDotProductAttention()
    output_no_mask, weights_no_mask = attention.forward(Q, K, V, mask=None)

    print("\n无掩码的注意力权重:")
    print(weights_no_mask[0])

    # 有掩码注意力
    output_masked, weights_masked = attention.forward(Q, K, V, mask=mask)

    print("\n有掩码的注意力权重（未来位置权重为0）:")
    print(weights_masked[0])

    print("\n解读:")
    print("  在解码器中，每个位置只能关注当前和之前的位置")
    print("  这确保了自回归生成：第t步只依赖前t-1步的信息")


def demo_attention_as_database_query():
    """演示：用数据库查询理解注意力机制"""
    print("\n" + "=" * 60)
    print("演示 5: 数据库查询类比")
    print("=" * 60)

    print("""
注意力机制可以类比为数据库查询：

1. Query (查询): "我想找关于'深度学习'的信息"
2. Key (键): 数据库中每条记录的索引/标签
3. Value (值): 数据库中每条记录的实际内容

Attention(Q,K,V) 的过程：
  Step 1: 用 Query 和所有 Key 计算相似度 → 得到每条记录的相关性分数
  Step 2: Softmax 归一化 → 转换为概率分布（注意力权重）
  Step 3: 用权重对 Value 加权求和 → 得到最终答案

例子：
  Query: "深度学习的应用"

  数据库:
    Key1: "计算机视觉"     Value1: "CNN用于图像分类..."      → 权重0.4
    Key2: "自然语言处理"   Value2: "Transformer用于翻译..."  → 权重0.5
    Key3: "推荐系统"       Value3: "协同过滤算法..."          → 权重0.1

  最终输出 = 0.4 * Value1 + 0.5 * Value2 + 0.1 * Value3
            (混合了多个相关信息，权重高的贡献更大)

为什么叫 Self-Attention？
  当 Q=K=V 都来自同一个输入序列时，就是"自注意力"
  比如句子"我爱深度学习"中，每个词既是 Query，也是 Key 和 Value
  这样每个词都能关注到句子中的其他词，理解上下文关系
    """)


def print_summary():
    """打印学习总结"""
    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. 核心公式
   Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

   - QK^T: 计算相似度
   - /sqrt(d_k): 缩放因子（防止梯度消失）
   - softmax: 归一化为概率分布
   - *V: 加权求和

2. 关键组件
   ✓ Scaled Dot-Product Attention: 基础注意力单元
   ✓ Multi-Head Attention: 并行多个注意力头，从不同子空间捕获特征
   ✓ Self-Attention: Q=K=V，序列内部的交互
   ✓ Masked Attention: 防止看到未来信息（解码器用）

3. 注意力的优势
   ✓ 并行计算（不像RNN需要顺序处理）
   ✓ 长距离依赖（任意两个位置直接连接）
   ✓ 可解释性（可视化注意力权重）

4. 在 Transformer 中的作用
   ✓ Encoder: Self-Attention（理解输入）
   ✓ Decoder: Masked Self-Attention（生成输出）+ Cross-Attention（关注编码器输出）

5. 现代应用
   ✓ NLP: BERT、GPT、T5
   ✓ CV: Vision Transformer (ViT)
   ✓ 多模态: CLIP、DALL-E

下一步：
  → 学习完整的 Transformer 架构（06_transformer_numpy.py）
  → 理解 Position Encoding（位置编码）
  → 看 PyTorch 版本（05_attention_pytorch.py）
    """)


if __name__ == "__main__":
    print("\n" + "🧠 " + "=" * 58)
    print("  Attention 机制 - NumPy手写实现")
    print("  理解 Transformer 的核心")
    print("=" * 60)

    # 运行所有演示
    demo_scaled_dot_product_attention()
    demo_multi_head_attention()
    demo_self_attention_with_meaning()
    demo_masked_attention()
    demo_attention_as_database_query()

    # 打印总结
    print_summary()

    print("\n✅ 所有演示完成！")
    print("📊 注意力可视化已保存到 attention_weights.png")
