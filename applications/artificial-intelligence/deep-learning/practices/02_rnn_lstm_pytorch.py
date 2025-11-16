"""
循环神经网络 (RNN/LSTM) - PyTorch 实现

对比 NumPy 版本：
- NumPy: 手写RNN/LSTM，理解门控机制
- PyTorch: 使用框架，GPU加速，工业实践

本文件内容：
1. PyTorch RNN/LSTM 基础组件
2. 完整的序列预测模型（时间序列）
3. GPU 训练加速
4. 训练可视化
5. 与 NumPy 版本性能对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time


# ==================== 1. PyTorch RNN/LSTM 基础组件 ====================
def demo_pytorch_rnn():
    """
    演示 PyTorch 的 RNN 操作

    ====================================================================
    🔑 PyTorch vs NumPy RNN
    ====================================================================

    NumPy 版本（手写循环）：
    ```python
    def forward(self, X):
        h = np.zeros((batch_size, hidden_size))
        for t in range(seq_len):
            h = np.tanh(np.dot(X[:, t, :], W_xh) + np.dot(h, W_hh) + b_h)
            # ...
    ```

    PyTorch 版本（一行）：
    ```python
    output, hidden = rnn(X)
    ```

    PyTorch 帮你做了什么？
    - 自动循环处理序列
    - 自动批量处理（batch）
    - 自动GPU加速
    - 自动计算梯度（BPTT - 时间反向传播）
    - 数值优化（更快更稳定）

    ====================================================================
    """
    print("=" * 70)
    print("1. PyTorch RNN 操作演示")
    print("=" * 70)

    # 创建输入序列（batch_size=2, seq_len=5, input_size=3）
    # PyTorch RNN 格式：(seq_len, batch_size, input_size) 或 (batch_size, seq_len, input_size)
    batch_size = 2
    seq_len = 5
    input_size = 3
    hidden_size = 4

    # batch_first=False: (seq_len, batch, input_size) - 默认格式
    X = torch.randn(seq_len, batch_size, input_size)
    print(f"\n输入序列 shape: {X.shape}")  # (5, 2, 3)

    # 创建 RNN 层
    rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=False)

    print(f"RNN hidden size: {hidden_size}")

    # 初始化隐藏状态（可选，默认为0）
    h0 = torch.zeros(1, batch_size, hidden_size)  # (num_layers, batch, hidden_size)

    # 前向传播
    output, hn = rnn(X, h0)

    print(f"\n输出 shape: {output.shape}")  # (seq_len, batch, hidden_size) = (5, 2, 4)
    print(f"最终隐藏状态 shape: {hn.shape}")  # (num_layers, batch, hidden_size) = (1, 2, 4)

    print("\n💡 PyTorch RNN 优势:")
    print("  - 一行代码完成所有时间步")
    print("  - 自动处理变长序列（pack_padded_sequence）")
    print("  - 自动计算梯度（BPTT）")
    print("  - GPU 加速（添加 .cuda()）")

    # batch_first=True 格式（更常用）
    print("\n" + "-" * 70)
    print("batch_first=True 格式（推荐）")
    print("-" * 70)

    X_batch_first = torch.randn(batch_size, seq_len, input_size)  # (2, 5, 3)
    rnn_batch_first = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                             batch_first=True)

    output, hn = rnn_batch_first(X_batch_first)
    print(f"输入 shape: {X_batch_first.shape}")  # (batch, seq_len, input_size)
    print(f"输出 shape: {output.shape}")  # (batch, seq_len, hidden_size)


def demo_pytorch_lstm():
    """演示 PyTorch 的 LSTM 操作"""
    print("\n" + "=" * 70)
    print("2. PyTorch LSTM 操作演示")
    print("=" * 70)

    batch_size = 2
    seq_len = 5
    input_size = 3
    hidden_size = 4

    # 创建输入
    X = torch.randn(batch_size, seq_len, input_size)
    print(f"\n输入序列 shape: {X.shape}")

    # 创建 LSTM 层
    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    # 初始化隐藏状态和细胞状态
    h0 = torch.zeros(1, batch_size, hidden_size)  # Hidden state
    c0 = torch.zeros(1, batch_size, hidden_size)  # Cell state

    # 前向传播
    output, (hn, cn) = lstm(X, (h0, c0))

    print(f"\n输出 shape: {output.shape}")  # (batch, seq_len, hidden_size)
    print(f"最终隐藏状态 shape: {hn.shape}")  # (num_layers, batch, hidden_size)
    print(f"最终细胞状态 shape: {cn.shape}")  # (num_layers, batch, hidden_size)

    print("\n💡 LSTM vs RNN:")
    print("  - LSTM 返回 (output, (hn, cn))，有两个状态")
    print("  - RNN 返回 (output, hn)，只有一个状态")
    print("  - LSTM 可以学习长期依赖")
    print("  - LSTM 参数量是 RNN 的 4 倍")

    # 多层 LSTM
    print("\n" + "-" * 70)
    print("多层 LSTM (Stacked LSTM)")
    print("-" * 70)

    lstm_stacked = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=2, batch_first=True)

    h0_stacked = torch.zeros(2, batch_size, hidden_size)  # 2 layers
    c0_stacked = torch.zeros(2, batch_size, hidden_size)

    output, (hn, cn) = lstm_stacked(X, (h0_stacked, c0_stacked))

    print(f"2层LSTM 输出 shape: {output.shape}")
    print(f"2层LSTM 隐藏状态 shape: {hn.shape}")  # (2, batch, hidden_size)


# ==================== 2. 完整的序列预测模型 ====================
class SequenceDataset(Dataset):
    """时间序列数据集"""

    def __init__(self, n_samples=1000, seq_len=20, noise=0.1):
        """
        生成正弦波序列用于预测

        任务：给定前 seq_len 个点，预测下一个点
        """
        self.X = []
        self.y = []

        for i in range(n_samples):
            start = np.random.uniform(0, 100)
            time = np.linspace(start, start + seq_len + 1, seq_len + 1)
            sequence = np.sin(time) + np.random.randn(seq_len + 1) * noise

            self.X.append(sequence[:-1])  # 输入: t=0 到 t=seq_len-1
            self.y.append(sequence[-1])   # 目标: t=seq_len

        self.X = torch.FloatTensor(self.X).unsqueeze(-1)  # (n_samples, seq_len, 1)
        self.y = torch.FloatTensor(self.y).unsqueeze(-1)  # (n_samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNModel(nn.Module):
    """
    Simple RNN 模型

    ====================================================================
    🔑 PyTorch RNN 模型结构
    ====================================================================

    网络结构：
    Input (batch, seq_len, input_size=1)
        ↓
    RNN (hidden_size=32)
        ↓
    Take last time step (batch, hidden_size)
        ↓
    Dropout (0.2)
        ↓
    Linear (hidden_size → 1)
        ↓
    Output (batch, 1)
    """

    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN 层
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # Dropout only for multi-layer
        )

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        返回: (batch_size, output_size)
        """
        # RNN forward
        # output: (batch, seq_len, hidden_size)
        # hn: (num_layers, batch, hidden_size)
        output, hn = self.rnn(x)

        # 取最后一个时间步的输出
        last_output = output[:, -1, :]  # (batch, hidden_size)

        # Dropout + Linear
        out = self.dropout(last_output)
        out = self.fc(out)  # (batch, output_size)

        return out


class LSTMModel(nn.Module):
    """
    LSTM 模型

    ====================================================================
    🔑 LSTM vs RNN
    ====================================================================

    相同点：
    - 都处理序列数据
    - 都有隐藏状态传递

    不同点：
    - LSTM 有细胞状态 (cell state)
    - LSTM 有三个门控机制
    - LSTM 可以学习长期依赖

    PyTorch 自动处理所有门控逻辑！
    """

    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        返回: (batch_size, output_size)
        """
        # LSTM forward
        # output: (batch, seq_len, hidden_size)
        # (hn, cn): (num_layers, batch, hidden_size)
        output, (hn, cn) = self.lstm(x)

        # 取最后一个时间步
        last_output = output[:, -1, :]

        out = self.dropout(last_output)
        out = self.fc(out)

        return out


# ==================== 3. 训练和评估 ====================
def train_one_epoch(model, device, train_loader, optimizer, criterion):
    """训练一个 epoch"""
    model.train()

    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # 前向传播
        output = model(data)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新权重
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, device, test_loader, criterion):
    """评估模型"""
    model.eval()

    test_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item()

            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    avg_loss = test_loss / len(test_loader)

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return avg_loss, predictions, targets


def train_rnn_lstm():
    """
    完整训练流程

    ====================================================================
    🔑 PyTorch 序列模型训练流程
    ====================================================================

    1. 准备数据
       - 序列数据：(batch, seq_len, input_size)
       - 使用 DataLoader

    2. 定义模型
       - 使用 nn.RNN 或 nn.LSTM
       - 提取最后时间步输出

    3. 训练技巧
       - 梯度裁剪：防止梯度爆炸
       - Dropout：防止过拟合
       - 多层堆叠：更强表达能力

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("3. 训练 RNN/LSTM 模型（时间序列预测）")
    print("=" * 70)

    # ========== 1. 检查 GPU ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    if torch.cuda.is_available():
        print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  ⚠️ 没有检测到 GPU，使用 CPU 训练")

    # ========== 2. 准备数据 ==========
    print("\n准备数据...")

    train_dataset = SequenceDataset(n_samples=1000, seq_len=20, noise=0.1)
    test_dataset = SequenceDataset(n_samples=200, seq_len=20, noise=0.1)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"  训练集: {len(train_dataset)} 个序列")
    print(f"  测试集: {len(test_dataset)} 个序列")
    print(f"  序列长度: {train_dataset.X.shape[1]}")
    print(f"  Batch size: {batch_size}")

    # ========== 3. 创建模型 ==========
    print("\n创建模型...")

    # RNN 模型
    rnn_model = RNNModel(input_size=1, hidden_size=32, output_size=1, num_layers=2)
    rnn_model = rnn_model.to(device)

    # LSTM 模型
    lstm_model = LSTMModel(input_size=1, hidden_size=32, output_size=1, num_layers=2)
    lstm_model = lstm_model.to(device)

    print(f"\nRNN 模型:")
    print(rnn_model)
    rnn_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"  参数量: {rnn_params:,}")

    print(f"\nLSTM 模型:")
    print(lstm_model)
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"  参数量: {lstm_params:,}")

    print(f"\n💡 LSTM 参数量约为 RNN 的 4 倍（因为有 3 个门）")

    # ========== 4. 定义损失函数和优化器 ==========
    criterion = nn.MSELoss()

    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # ========== 5. 训练 ==========
    print("\n开始训练...")
    n_epochs = 50

    rnn_train_losses = []
    rnn_test_losses = []
    lstm_train_losses = []
    lstm_test_losses = []

    # 训练 RNN
    print(f"\n{'='*70}")
    print("训练 RNN 模型...")
    print(f"{'='*70}")

    rnn_start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(rnn_model, device, train_loader,
                                     rnn_optimizer, criterion)
        test_loss, _, _ = evaluate(rnn_model, device, test_loader, criterion)

        rnn_train_losses.append(train_loss)
        rnn_test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    rnn_time = time.time() - rnn_start_time
    print(f"\nRNN 训练完成！耗时: {rnn_time:.2f} 秒")

    # 训练 LSTM
    print(f"\n{'='*70}")
    print("训练 LSTM 模型...")
    print(f"{'='*70}")

    lstm_start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(lstm_model, device, train_loader,
                                     lstm_optimizer, criterion)
        test_loss, _, _ = evaluate(lstm_model, device, test_loader, criterion)

        lstm_train_losses.append(train_loss)
        lstm_test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    lstm_time = time.time() - lstm_start_time
    print(f"\nLSTM 训练完成！耗时: {lstm_time:.2f} 秒")

    # ========== 6. 最终评估 ==========
    _, rnn_preds, rnn_targets = evaluate(rnn_model, device, test_loader, criterion)
    _, lstm_preds, lstm_targets = evaluate(lstm_model, device, test_loader, criterion)

    print(f"\n{'='*70}")
    print("最终结果对比")
    print(f"{'='*70}")
    print(f"RNN  - Test Loss: {rnn_test_losses[-1]:.6f} | 训练时间: {rnn_time:.2f}s")
    print(f"LSTM - Test Loss: {lstm_test_losses[-1]:.6f} | 训练时间: {lstm_time:.2f}s")

    # ========== 7. 可视化 ==========
    visualize_training(n_epochs, rnn_train_losses, rnn_test_losses,
                      lstm_train_losses, lstm_test_losses,
                      rnn_preds, lstm_preds, rnn_targets, test_dataset)

    return rnn_model, lstm_model


def visualize_training(n_epochs, rnn_train_losses, rnn_test_losses,
                      lstm_train_losses, lstm_test_losses,
                      rnn_preds, lstm_preds, targets, test_dataset):
    """可视化训练结果"""
    print("\n可视化训练结果...")

    fig = plt.figure(figsize=(16, 10))

    # 1. 训练损失曲线
    ax1 = plt.subplot(2, 3, 1)
    epochs_range = range(1, n_epochs + 1)
    ax1.plot(epochs_range, rnn_train_losses, 'b-', label='RNN Train', linewidth=2)
    ax1.plot(epochs_range, rnn_test_losses, 'b--', label='RNN Test', linewidth=2)
    ax1.plot(epochs_range, lstm_train_losses, 'r-', label='LSTM Train', linewidth=2)
    ax1.plot(epochs_range, lstm_test_losses, 'r--', label='LSTM Test', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_title('Training Loss: RNN vs LSTM', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2. 测试损失对比
    ax2 = plt.subplot(2, 3, 2)
    models = ['RNN', 'LSTM']
    final_losses = [rnn_test_losses[-1], lstm_test_losses[-1]]
    colors = ['#3498db', '#e74c3c']
    bars = ax2.bar(models, final_losses, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Test Loss (MSE)', fontsize=11)
    ax2.set_title('Final Test Loss Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.6f}', ha='center', va='bottom', fontsize=10)

    # 3. RNN 预测 vs 真实值（散点图）
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(targets, rnn_preds, alpha=0.5, s=20)
    ax3.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax3.set_xlabel('True Values', fontsize=11)
    ax3.set_ylabel('RNN Predictions', fontsize=11)
    ax3.set_title('RNN: Predictions vs Truth', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # 4. LSTM 预测 vs 真实值（散点图）
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(targets, lstm_preds, alpha=0.5, s=20, color='red')
    ax4.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
            'b--', lw=2, label='Perfect Prediction')
    ax4.set_xlabel('True Values', fontsize=11)
    ax4.set_ylabel('LSTM Predictions', fontsize=11)
    ax4.set_title('LSTM: Predictions vs Truth', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # 5. 示例序列预测（RNN）
    ax5 = plt.subplot(2, 3, 5)
    n_show = 5
    for i in range(n_show):
        seq = test_dataset.X[i, :, 0].numpy()
        ax5.plot(seq, alpha=0.6, linewidth=2)
        # 真实值（红色星星）
        ax5.scatter(len(seq), targets[i], color='red', s=150, marker='*',
                   zorder=5, edgecolors='black', linewidths=1)
        # RNN预测（蓝色圆圈）
        ax5.scatter(len(seq), rnn_preds[i], color='blue', s=80, marker='o',
                   zorder=5, edgecolors='black', linewidths=1)

    ax5.set_xlabel('Time Step', fontsize=11)
    ax5.set_ylabel('Value', fontsize=11)
    ax5.set_title('RNN Predictions (Red=True, Blue=Pred)', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)

    # 6. 示例序列预测（LSTM）
    ax6 = plt.subplot(2, 3, 6)
    for i in range(n_show):
        seq = test_dataset.X[i, :, 0].numpy()
        ax6.plot(seq, alpha=0.6, linewidth=2)
        # 真实值（红色星星）
        ax6.scatter(len(seq), targets[i], color='red', s=150, marker='*',
                   zorder=5, edgecolors='black', linewidths=1)
        # LSTM预测（绿色圆圈）
        ax6.scatter(len(seq), lstm_preds[i], color='green', s=80, marker='o',
                   zorder=5, edgecolors='black', linewidths=1)

    ax6.set_xlabel('Time Step', fontsize=11)
    ax6.set_ylabel('Value', fontsize=11)
    ax6.set_title('LSTM Predictions (Red=True, Green=Pred)', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('rnn_lstm_pytorch_training.png', dpi=100, bbox_inches='tight')
    print("📊 训练结果已保存: rnn_lstm_pytorch_training.png")
    plt.close()


# ==================== 4. PyTorch vs NumPy 对比 ====================
def compare_pytorch_vs_numpy():
    """
    对比 PyTorch 和 NumPy 版本

    ====================================================================
    🔑 PyTorch vs NumPy
    ====================================================================

    NumPy 版本：
    ✅ 优点：
      - 理解RNN/LSTM内部机制
      - 手写门控逻辑，掌握细节
      - 不依赖深度学习框架

    ❌ 缺点：
      - 代码量大（需要手写BPTT）
      - 速度慢（无GPU加速）
      - 难以处理复杂序列（变长、padding）
      - 数值不稳定

    PyTorch 版本：
    ✅ 优点：
      - 代码简洁（几行搞定）
      - GPU 加速（快10-100倍）
      - 自动微分（不需要手写BPTT）
      - 内置优化（CuDNN加速）
      - 工业界标准

    ❌ 缺点：
      - 框架黑盒（不知道内部细节）
      - 需要学习新API

    建议：
    - 学习阶段：先看 NumPy 版本（理解原理）
    - 实践阶段：用 PyTorch 版本（实际应用）

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("4. PyTorch vs NumPy 对比")
    print("=" * 70)

    print("""
性能对比（时间序列预测）：

+----------------+------------------+------------------+
|     指标       |   NumPy 版本     |  PyTorch 版本    |
+----------------+------------------+------------------+
| 代码量         | ~400 行          | ~150 行          |
| 训练时间       | ~2 分钟 (CPU)    | ~10 秒 (GPU)     |
| 测试准确率     | 好               | 更好             |
| GPU 支持       | ❌               | ✅               |
| 自动微分       | ❌ (手写BPTT)    | ✅               |
| 变长序列       | 困难             | ✅ (pack/pad)    |
| 工业应用       | ❌               | ✅               |
+----------------+------------------+------------------+

代码对比：

NumPy 版本（复杂）：
```python
# 手写LSTM前向传播
def forward(self, X):
    for t in range(seq_len):
        combined = np.concatenate([x_t, h], axis=1)

        # 手动计算每个门
        f_t = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)
        i_t = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        o_t = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        c_tilde = np.tanh(np.dot(combined, self.W_c) + self.b_c)

        # 手动更新状态
        c = f_t * c + i_t * c_tilde
        h = o_t * np.tanh(c)
        # ...

# 手写BPTT反向传播（更复杂！）
# ... 100+ 行梯度计算代码
```

PyTorch 版本（简洁）：
```python
# 定义模型
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return self.fc(output[:, -1, :])

# 训练（自动BPTT！）
output = model(data)
loss = criterion(output, target)
loss.backward()          # ← 自动BPTT！
optimizer.step()         # ← 自动更新！
```

总结：
- 学习原理 → 用 NumPy（理解门控机制）
- 实际应用 → 用 PyTorch（工业标准）
- 两者结合 → 最佳理解！
    """)


# ==================== 5. 主程序 ====================
def main():
    print("=" * 70)
    print("循环神经网络 (RNN/LSTM) - PyTorch 实现")
    print("=" * 70)

    # 1. PyTorch 基础组件
    demo_pytorch_rnn()
    demo_pytorch_lstm()

    # 2. 训练完整模型
    rnn_model, lstm_model = train_rnn_lstm()

    # 3. 对比 PyTorch vs NumPy
    compare_pytorch_vs_numpy()

    # 4. 总结
    print("\n" + "=" * 70)
    print("✅ 核心要点总结")
    print("=" * 70)
    print("""
1. PyTorch RNN/LSTM 基础组件

   RNN层：
   nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

   LSTM层：
   nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

   GRU层：
   nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

2. 定义序列模型（继承 nn.Module）

   class MyRNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.rnn = nn.RNN(input_size=1, hidden_size=32)
           self.fc = nn.Linear(32, 1)

       def forward(self, x):
           output, hn = self.rnn(x)
           # 取最后时间步
           last_output = output[:, -1, :]
           return self.fc(last_output)

3. 序列数据格式

   batch_first=True:  (batch_size, seq_len, input_size)
   batch_first=False: (seq_len, batch_size, input_size)

   推荐使用 batch_first=True（更直观）

4. 训练技巧

   # 梯度裁剪（防止梯度爆炸）
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

   # Dropout（防止过拟合）
   nn.LSTM(..., dropout=0.2)  # 仅用于多层
   nn.Dropout(0.2)  # 在LSTM后添加

   # 多层堆叠
   nn.LSTM(..., num_layers=2)

5. GPU 加速

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   data = data.to(device)

   速度提升：CPU 2分钟 → GPU 10秒（10-20倍）

6. RNN vs LSTM

   RNN:
   - 简单，参数少
   - 快速训练
   - 短序列表现好
   - 长序列有梯度消失问题

   LSTM:
   - 复杂，参数多（4倍）
   - 训练较慢
   - 长序列表现好
   - 有细胞状态和门控机制

7. PyTorch vs NumPy

   NumPy:
   - 理解原理（手写门控）
   - 代码量大
   - 速度慢

   PyTorch:
   - 工业实践（自动微分）
   - 代码简洁
   - 速度快10-100倍

8. 实践建议

   学习路径：
   1. 先看 NumPy 版本（理解LSTM门控机制）
   2. 再看 PyTorch 版本（学习框架）
   3. 对比两个版本（理解框架做了什么）

   实际工作：
   - 100% 用 PyTorch（或 TensorFlow）
   - NumPy 只用于理解原理

9. 序列建模应用

   - 时间序列预测：股价、天气
   - 自然语言处理：文本生成、翻译
   - 推荐系统：用户行为序列建模
   - 语音识别：音频序列
   - 视频分析：帧序列

10. 下一步学习

    - Attention机制（解决长序列问题）
    - Transformer（取代RNN的现代架构）
    - GRU（LSTM的简化版本）
    - Bidirectional RNN（双向处理）
    - Seq2Seq（编码器-解码器）
    """)


if __name__ == "__main__":
    main()

    print("\n💡 练习建议:")
    print("  1. 尝试不同的 hidden_size 和 num_layers")
    print("  2. 在真实时间序列数据上训练（股价、天气）")
    print("  3. 实现双向 LSTM（bidirectional=True）")
    print("  4. 比较 RNN、LSTM、GRU 的性能")
    print("  5. 理解为什么LSTM能处理长序列")
    print("  6. 思考：如何用RNN建模用户行为序列？")
