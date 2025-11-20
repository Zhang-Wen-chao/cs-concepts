"""
Attention 机制 - PyTorch实现
实际应用：序列到序列任务 + GPU加速

作者: Zhang Wenchao
日期: 2025-11-20
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


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力 - PyTorch版本

    使用 PyTorch 的张量操作和自动求导
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q: (batch_size, num_heads, seq_len, d_k)
            K: (batch_size, num_heads, seq_len, d_k)
            V: (batch_size, num_heads, seq_len, d_v)
            mask: (batch_size, 1, 1, seq_len) 或 None

        返回:
            output: (batch_size, num_heads, seq_len, d_v)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)

        # 1. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

        # 2. 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax 得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. 加权求和
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力 - PyTorch版本

    使用 nn.Linear 实现线性投影，支持自动求导和GPU加速
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层（自动初始化）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        拆分为多个头

        输入: (batch_size, seq_len, d_model)
        输出: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        """
        合并多个头

        输入: (batch_size, num_heads, seq_len, d_k)
        输出: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        参数:
            Q, K, V: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) 或 None

        返回:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # 1. 线性投影
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. 拆分为多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. 计算注意力
        x, attention_weights = self.attention(Q, K, V, mask)

        # 4. 合并多个头
        x = self.combine_heads(x)

        # 5. 最终投影
        output = self.W_O(x)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise 前馈网络

    Transformer 的第二个子层：两层全连接网络
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Transformer Encoder 层

    包含：
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len)

        返回:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # 1. Multi-Head Self-Attention + Residual + Norm
        attn_output, attention_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Feed-Forward + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x, attention_weights


class PositionalEncoding(nn.Module):
    """位置编码

    由于 Attention 机制没有位置信息，需要手动添加位置编码
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SimpleTransformerEncoder(nn.Module):
    """简单的 Transformer Encoder（用于序列分类）

    应用：文本分类、情感分析等
    """

    def __init__(self, vocab_size, d_model=128, num_heads=8, num_layers=3,
                 d_ff=512, max_len=100, num_classes=2, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x: (batch_size, seq_len) - 词ID序列
            mask: (batch_size, 1, 1, seq_len)

        返回:
            output: (batch_size, num_classes)
            all_attention_weights: list of attention weights
        """
        # 1. Embedding + Positional Encoding
        x = self.embedding(x) * np.sqrt(self.d_model)  # 缩放
        x = self.pos_encoding(x)

        # 2. 通过所有 Encoder 层
        all_attention_weights = []
        for encoder_layer in self.encoder_layers:
            x, attn_weights = encoder_layer(x, mask)
            all_attention_weights.append(attn_weights)

        # 3. 池化：取第一个token的表示（类似BERT的[CLS]）
        x = x[:, 0, :]

        # 4. 分类
        output = self.fc(self.dropout(x))

        return output, all_attention_weights


# ============ 实际应用：序列分类任务 ============

class SyntheticSequenceDataset(Dataset):
    """合成序列分类数据集

    任务：判断序列是否包含特定模式
    - 类别0: 序列中没有连续的大数字 (>50)
    - 类别1: 序列中有连续的大数字
    """

    def __init__(self, num_samples=1000, seq_len=20, vocab_size=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.data = []
        self.labels = []

        for _ in range(num_samples):
            seq = np.random.randint(1, vocab_size, size=seq_len)

            # 规则：如果有连续3个大于50的数字，标签为1
            has_pattern = False
            for i in range(seq_len - 2):
                if seq[i] > 50 and seq[i+1] > 50 and seq[i+2] > 50:
                    has_pattern = True
                    break

            self.data.append(seq)
            self.labels.append(1 if has_pattern else 0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])


def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()

            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()

                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return history


def visualize_attention(model, data_loader, device, save_path='attention_heatmap.png'):
    """可视化注意力权重"""
    model.eval()

    # 获取一个batch
    batch_x, batch_y = next(iter(data_loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # 前向传播获取注意力权重
    with torch.no_grad():
        outputs, all_attention_weights = model(batch_x)

    # 取第一个样本的第一个头的注意力权重
    attention = all_attention_weights[0][0, 0].cpu().numpy()  # 第一层，第一个样本，第一个头

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention, cmap='viridis', aspect='auto')

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title('Multi-Head Attention Weights (Layer 1, Head 1)')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"注意力可视化已保存到 {save_path}")
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy曲线
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练历史已保存到 {save_path}")
    plt.close()


def compare_cpu_gpu_speed():
    """对比CPU和GPU的速度"""
    print("\n" + "=" * 60)
    print("GPU vs CPU 速度对比")
    print("=" * 60)

    d_model = 512
    num_heads = 8
    batch_size = 32
    seq_len = 100

    # 创建模型和数据
    model = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # CPU测试
    model_cpu = model.cpu()
    x_cpu = x.cpu()

    start = time.time()
    for _ in range(100):
        _ = model_cpu(x_cpu, x_cpu, x_cpu)
    cpu_time = time.time() - start

    print(f"CPU 时间 (100次前向传播): {cpu_time:.4f}秒")

    # GPU测试
    if torch.cuda.is_available():
        model_gpu = model.cuda()
        x_gpu = x.cuda()

        # 预热
        for _ in range(10):
            _ = model_gpu(x_gpu, x_gpu, x_gpu)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = model_gpu(x_gpu, x_gpu, x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"GPU 时间 (100次前向传播): {gpu_time:.4f}秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU 不可用")


def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  Attention 机制 - PyTorch实现")
    print("  实战：序列分类 + GPU加速")
    print("=" * 60)

    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 速度对比
    compare_cpu_gpu_speed()

    # 创建数据集
    print("\n" + "=" * 60)
    print("创建合成数据集")
    print("=" * 60)

    train_dataset = SyntheticSequenceDataset(num_samples=2000, seq_len=20, vocab_size=100)
    val_dataset = SyntheticSequenceDataset(num_samples=500, seq_len=20, vocab_size=100)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建模型
    print("\n" + "=" * 60)
    print("创建 Transformer Encoder 模型")
    print("=" * 60)

    model = SimpleTransformerEncoder(
        vocab_size=100,
        d_model=128,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        max_len=20,
        num_classes=2,
        dropout=0.1
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    history = train_model(model, train_loader, val_loader, device, num_epochs=20)

    # 可视化
    print("\n" + "=" * 60)
    print("生成可视化")
    print("=" * 60)

    plot_training_history(history)
    visualize_attention(model, val_loader, device)

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. PyTorch vs NumPy 的区别
   ✓ 自动求导: 不需要手写反向传播
   ✓ GPU加速: .to(device) 即可使用GPU
   ✓ 模块化: nn.Module 封装模型
   ✓ 优化器: torch.optim 自动更新参数

2. Transformer Encoder 组件
   ✓ Positional Encoding: 添加位置信息
   ✓ Multi-Head Attention: 多角度关注
   ✓ Feed-Forward Network: 非线性变换
   ✓ Layer Normalization: 稳定训练

3. 工业实践技巧
   ✓ Dropout: 防止过拟合
   ✓ Residual Connection: 缓解梯度消失
   ✓ Layer Norm: 加速收敛
   ✓ Learning Rate Scheduling: 提升性能

4. 下一步
   → 完整的 Transformer（Encoder + Decoder）
   → 预训练模型（BERT、GPT）
   → 实际NLP任务（文本分类、翻译）
    """)

    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
