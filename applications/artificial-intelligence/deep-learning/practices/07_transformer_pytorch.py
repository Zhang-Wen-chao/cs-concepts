"""
Transformer 架构 - PyTorch实现
完整的 Encoder-Decoder 结构 + 机器翻译实战

作者: Zhang Wenchao
日期: 2025-11-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检测设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力（与 06 相同）"""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力（支持 Self-Attention 和 Cross-Attention）"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        x, attention_weights = self.attention(Q, K, V, mask)
        x = self.combine_heads(x)
        output = self.W_O(x)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Transformer Encoder 层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder 层（新增！）

    包含三个子层：
    1. Masked Self-Attention（只能看到当前和之前的词）
    2. Cross-Attention（关注 Encoder 输出）
    3. Feed-Forward Network
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x: (batch, tgt_seq_len, d_model) - Decoder 输入
            encoder_output: (batch, src_seq_len, d_model) - Encoder 输出
            src_mask: Encoder 的 padding mask
            tgt_mask: Decoder 的 look-ahead mask（下三角）

        返回:
            output: (batch, tgt_seq_len, d_model)
        """
        # 1. Masked Self-Attention + Residual + Norm
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # 2. Cross-Attention + Residual + Norm
        # Query 来自 Decoder，Key 和 Value 来自 Encoder
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # 3. FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder（堆叠多个 EncoderLayer）"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder（堆叠多个 DecoderLayer）"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        output = self.fc(x)
        return output


class Transformer(nn.Module):
    """完整的 Transformer 模型（Encoder-Decoder）"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=100, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )

    def create_look_ahead_mask(self, size):
        """创建 Decoder 的 look-ahead mask（下三角矩阵）"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask  # 反转：下三角为 True（允许），上三角为 False（禁止）

    def forward(self, src, tgt):
        """
        参数:
            src: (batch, src_seq_len) - 源序列
            tgt: (batch, tgt_seq_len) - 目标序列

        返回:
            output: (batch, tgt_seq_len, tgt_vocab_size)
        """
        # 1. Encoder
        encoder_output = self.encoder(src)

        # 2. 创建 Decoder 的 look-ahead mask
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.create_look_ahead_mask(tgt_seq_len).to(tgt.device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # 3. Decoder
        output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask)

        return output


# ============ 实际应用：简单的序列到序列任务 ============

class Seq2SeqDataset(Dataset):
    """简单的序列到序列数据集

    任务：数字序列反转
    例子: [1, 2, 3, 4] -> [4, 3, 2, 1]
    """

    def __init__(self, num_samples=1000, seq_len=10, vocab_size=50):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.data = []
        self.targets = []

        # 特殊标记
        self.PAD_IDX = 0
        self.BOS_IDX = 1  # Begin of Sequence
        self.EOS_IDX = 2  # End of Sequence

        for _ in range(num_samples):
            # 生成源序列（从3开始，避免特殊标记）
            src = np.random.randint(3, vocab_size, size=seq_len)

            # 目标序列：反转 + 添加 BOS 和 EOS
            tgt_input = np.concatenate([[self.BOS_IDX], src[::-1]])  # [BOS, 反转序列]
            tgt_output = np.concatenate([src[::-1], [self.EOS_IDX]])  # [反转序列, EOS]

            self.data.append((src, tgt_input))
            self.targets.append(tgt_output)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src, tgt_input = self.data[idx]
        tgt_output = self.targets[idx]
        return (
            torch.LongTensor(src),
            torch.LongTensor(tgt_input),
            torch.LongTensor(tgt_output)
        )


def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.0001):
    """训练模型"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for src, tgt_input, tgt_output in train_loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            optimizer.zero_grad()

            # 前向传播
            output = model(src, tgt_input)  # (batch, tgt_seq_len, vocab_size)

            # 计算损失
            output_flat = output.view(-1, output.size(-1))
            tgt_flat = tgt_output.view(-1)
            loss = criterion(output_flat, tgt_flat)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = output.max(-1)
            mask = tgt_output != 0  # 忽略 padding
            train_correct += (predicted == tgt_output).masked_select(mask).sum().item()
            train_total += mask.sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for src, tgt_input, tgt_output in val_loader:
                src = src.to(device)
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)

                output = model(src, tgt_input)

                output_flat = output.view(-1, output.size(-1))
                tgt_flat = tgt_output.view(-1)
                loss = criterion(output_flat, tgt_flat)

                val_loss += loss.item()
                _, predicted = output.max(-1)
                mask = tgt_output != 0
                val_correct += (predicted == tgt_output).masked_select(mask).sum().item()
                val_total += mask.sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_transformer.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n早停触发！在 epoch {epoch+1}')
                # 加载最佳模型
                model.load_state_dict(torch.load('best_transformer.pth'))
                break

        if (epoch + 1) % 10 == 0:  # 每10轮打印一次
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')

    return history


def test_inference(model, dataset, device, num_examples=5):
    """测试推理（逐词生成）"""
    model.eval()

    print("\n" + "=" * 60)
    print("推理测试（序列反转）")
    print("=" * 60)

    BOS_IDX = dataset.BOS_IDX
    EOS_IDX = dataset.EOS_IDX
    max_len = dataset.seq_len + 2

    for i in range(num_examples):
        src, _, tgt_output = dataset[i]
        src = src.unsqueeze(0).to(device)  # (1, seq_len)

        # 编码源序列
        encoder_output = model.encoder(src)

        # 初始化解码器输入（只有 BOS）
        tgt = torch.LongTensor([[BOS_IDX]]).to(device)

        # 逐词生成
        for _ in range(max_len):
            tgt_mask = model.create_look_ahead_mask(tgt.size(1)).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            output = model.decoder(tgt, encoder_output, tgt_mask=tgt_mask)
            next_token = output[:, -1, :].argmax(-1)  # 取最后一个位置的预测

            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == EOS_IDX:
                break

        # 打印结果
        src_seq = src.squeeze().cpu().numpy()
        pred_seq = tgt.squeeze().cpu().numpy()[1:-1]  # 去掉 BOS 和 EOS
        target_seq = tgt_output.numpy()

        print(f"\n例子 {i+1}:")
        print(f"  输入: {src_seq}")
        print(f"  预测: {pred_seq}")
        print(f"  目标: {target_seq[:-1]}")  # 去掉 EOS
        print(f"  正确: {'✓' if np.array_equal(pred_seq, target_seq[:-1]) else '✗'}")


def plot_training_history(history, save_path='transformer_training_history.png'):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n训练历史已保存到 {save_path}")
    plt.close()


def visualize_look_ahead_mask():
    """可视化 look-ahead mask"""
    seq_len = 8
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = (~mask).float().numpy()  # 反转并转换为浮点数

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mask, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Look-Ahead Mask (Decoder)\nGreen=Allowed, Red=Masked')

    for i in range(seq_len):
        for j in range(seq_len):
            text = '✓' if mask[i, j] == 1 else '✗'
            color = 'black' if mask[i, j] == 1 else 'white'
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=12)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('transformer_look_ahead_mask.png', dpi=150, bbox_inches='tight')
    print("Look-Ahead Mask visualization saved to transformer_look_ahead_mask.png")
    plt.close()


def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  Transformer 架构 - PyTorch实现")
    print("  实战：序列到序列（Seq2Seq）任务")
    print("=" * 60)

    # 检查GPU
    print(f"\n使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 创建数据集
    print("\n" + "=" * 60)
    print("创建数据集（序列反转任务）")
    print("=" * 60)

    vocab_size = 30  # 降低词汇表大小，更容易学习
    seq_len = 6  # 降低序列长度，更容易学习

    train_dataset = Seq2SeqDataset(num_samples=10000, seq_len=seq_len, vocab_size=vocab_size)  # 大幅增加数据
    val_dataset = Seq2SeqDataset(num_samples=2000, seq_len=seq_len, vocab_size=vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"任务: 序列反转（如 [5,8,3] -> [3,8,5]）")

    # 创建模型
    print("\n" + "=" * 60)
    print("创建 Transformer 模型")
    print("=" * 60)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,  # 降低模型容量，防止过拟合
        num_heads=4,  # 减少注意力头数
        num_layers=2,  # 减少层数
        d_ff=512,  # 减少FFN维度
        max_len=seq_len + 2,
        dropout=0.2  # 增加dropout，防止过拟合
    ).to(DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 可视化 look-ahead mask
    print("\n" + "=" * 60)
    print("可视化 Look-Ahead Mask")
    print("=" * 60)
    visualize_look_ahead_mask()

    # 训练模型
    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    history = train_model(model, train_loader, val_loader, DEVICE, num_epochs=100, lr=0.001)  # 更多轮次，但有早停

    # 绘制训练历史
    plot_training_history(history)

    # 测试推理
    test_inference(model, val_dataset, DEVICE, num_examples=10)  # 测试更多例子

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. Transformer 核心组件
   ✓ Encoder: 编码源序列
   ✓ Decoder: 生成目标序列（自回归）
   ✓ Cross-Attention: Decoder 关注 Encoder 输出

2. Decoder 的关键设计
   ✓ Masked Self-Attention: 防止"作弊"（看到未来信息）
   ✓ Look-Ahead Mask: 下三角矩阵掩码
   ✓ 推理时逐词生成（Auto-regressive）

3. 训练 vs 推理的区别
   ✓ 训练: 并行计算（Teacher Forcing，给完整目标序列）
   ✓ 推理: 逐词生成（从 BOS 开始，直到 EOS）

4. 应用场景
   ✓ 机器翻译: 英文 → 中文
   ✓ 文本摘要: 长文本 → 摘要
   ✓ 对话系统: 问题 → 回答
   ✓ 代码生成: 描述 → 代码

5. 与 Attention 的关系
   ✓ Attention 是 Transformer 的核心组件
   ✓ Transformer = Encoder + Decoder + Cross-Attention
   ✓ BERT 只用 Encoder，GPT 只用 Decoder

6. 下一步
   → 预训练模型（BERT、GPT）
   → 实际 NLP 任务（分类、翻译、摘要）
   → 推荐系统中的双塔模型
    """)

    print("\n✅ 训练完成！")
    print("\n提示: Transformer 是现代 NLP/推荐系统的基础")
    print("      接下来可以学习如何应用预训练模型或开始推荐系统实践")


if __name__ == "__main__":
    main()
