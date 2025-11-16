"""
优化与正则化技术 - PyTorch 实现

对比 NumPy 版本：
- NumPy: 手写优化器，理解更新规则
- PyTorch: 使用 torch.optim，GPU加速，工业实践

本文件内容：
1. PyTorch 优化器 (SGD, Momentum, RMSprop, Adam)
2. 学习率调度 (LR Scheduling)
3. 正则化技术 (Dropout, BatchNorm, L2)
4. 梯度裁剪 (Gradient Clipping)
5. 权重初始化 (Weight Initialization)
6. 完整训练示例
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
import time


# ==================== 1. PyTorch 优化器 ====================
def demo_pytorch_optimizers():
    """
    演示 PyTorch 的各种优化器

    ====================================================================
    🔑 PyTorch vs NumPy 优化器
    ====================================================================

    NumPy 版本（手写更新规则）：
    ```python
    class Adam:
        def update(self, params, grads):
            # 手动计算一阶矩、二阶矩
            m = beta1 * m + (1-beta1) * grad
            v = beta2 * v + (1-beta2) * grad**2
            # 手动偏差修正
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            # 手动更新
            param -= lr * m_hat / (sqrt(v_hat) + eps)
    ```

    PyTorch 版本（一行）：
    ```python
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练循环中
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # ← 自动更新所有参数！
    ```

    PyTorch 帮你做了什么？
    - 自动管理所有参数
    - 自动更新（内置最优实现）
    - GPU 加速
    - 内置所有现代优化器

    ====================================================================
    """
    print("=" * 70)
    print("1. PyTorch 优化器演示")
    print("=" * 70)

    # 创建一个简单模型
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    # ========== SGD ==========
    print("\n" + "-" * 70)
    print("SGD (Stochastic Gradient Descent)")
    print("-" * 70)
    print("""
更新规则：
    θ = θ - learning_rate × gradient

特点：
    - 最基础的优化器
    - 固定学习率
    - 可能震荡，收敛慢
    """)

    sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(f"创建 SGD: {sgd_optimizer}")

    # ========== SGD + Momentum ==========
    print("\n" + "-" * 70)
    print("SGD with Momentum")
    print("-" * 70)
    print("""
更新规则：
    velocity = momentum × velocity - learning_rate × gradient
    θ = θ + velocity

特点：
    - 累积历史梯度（惯性）
    - 加速收敛
    - 可以冲过小坑（局部最小值）
    - momentum 通常取 0.9
    """)

    momentum_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(f"创建 SGD+Momentum: {momentum_optimizer}")

    # ========== RMSprop ==========
    print("\n" + "-" * 70)
    print("RMSprop")
    print("-" * 70)
    print("""
更新规则：
    cache = decay × cache + (1-decay) × gradient²
    θ = θ - learning_rate × gradient / (√cache + ε)

特点：
    - 自适应学习率
    - 对频繁变化的参数用小学习率
    - 对稀疏变化的参数用大学习率
    """)

    rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
    print(f"创建 RMSprop: {rmsprop_optimizer}")

    # ========== Adam ==========
    print("\n" + "-" * 70)
    print("Adam (Adaptive Moment Estimation)")
    print("-" * 70)
    print("""
更新规则：
    m = β₁ × m + (1-β₁) × gradient       (一阶矩：动量)
    v = β₂ × v + (1-β₂) × gradient²      (二阶矩：自适应)
    m_hat = m / (1 - β₁ᵗ)                (偏差修正)
    v_hat = v / (1 - β₂ᵗ)
    θ = θ - lr × m_hat / (√v_hat + ε)

特点：
    - Momentum + RMSprop 结合
    - 自适应学习率 + 动量加速
    - 默认参数通常就很好 (β₁=0.9, β₂=0.999)
    - 最流行的优化器！
    """)

    adam_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    print(f"创建 Adam: {adam_optimizer}")

    # ========== AdamW ==========
    print("\n" + "-" * 70)
    print("AdamW (Adam with Weight Decay)")
    print("-" * 70)
    print("""
特点：
    - Adam 的改进版
    - 正确实现了权重衰减（L2正则化）
    - 在 Transformer 等模型中表现更好
    - 现代深度学习的首选优化器
    """)

    adamw_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    print(f"创建 AdamW: {adamw_optimizer}")

    print("\n💡 选择建议:")
    print("  - 快速原型: Adam (默认选择)")
    print("  - 最佳性能: AdamW (Transformer 等)")
    print("  - 简单任务: SGD + Momentum")
    print("  - 研究对比: 都试试，看哪个好")


def compare_optimizers_pytorch():
    """对比不同优化器在实际训练中的表现"""
    print("\n" + "=" * 70)
    print("2. 优化器性能对比（实际训练）")
    print("=" * 70)

    # ========== 准备数据 ==========
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 转为 PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"\n数据集:")
    print(f"  训练集: {len(X_train)} samples")
    print(f"  测试集: {len(X_test)} samples")

    # ========== 定义模型 ==========
    def create_model():
        return nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    # ========== 训练不同优化器 ==========
    optimizers_config = {
        'SGD': {'class': optim.SGD, 'kwargs': {'lr': 0.1}},
        'SGD+Momentum': {'class': optim.SGD, 'kwargs': {'lr': 0.1, 'momentum': 0.9}},
        'RMSprop': {'class': optim.RMSprop, 'kwargs': {'lr': 0.01}},
        'Adam': {'class': optim.Adam, 'kwargs': {'lr': 0.01}},
        'AdamW': {'class': optim.AdamW, 'kwargs': {'lr': 0.01, 'weight_decay': 0.01}},
    }

    criterion = nn.BCELoss()
    n_epochs = 50

    results = {}

    for name, config in optimizers_config.items():
        print(f"\n训练 {name}...")

        model = create_model()
        optimizer = config['class'](model.parameters(), **config['kwargs'])

        train_losses = []
        test_losses = []

        for epoch in range(n_epochs):
            # 训练
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            # 测试
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_loss = criterion(test_output, y_test)
                test_losses.append(test_loss.item())

        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_test_loss': test_losses[-1]
        }

        print(f"  最终测试损失: {test_losses[-1]:.4f}")

    # ========== 可视化 ==========
    visualize_optimizer_comparison(results, n_epochs)

    return results


def visualize_optimizer_comparison(results, n_epochs):
    """可视化优化器对比"""
    print("\n可视化优化器对比...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    epochs_range = range(1, n_epochs + 1)

    # Plot 1: 训练损失
    for (name, data), color in zip(results.items(), colors):
        axes[0].plot(epochs_range, data['train_losses'], label=name,
                    color=color, linewidth=2, alpha=0.8)

    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Training Loss', fontsize=11)
    axes[0].set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Plot 2: 测试损失
    for (name, data), color in zip(results.items(), colors):
        axes[1].plot(epochs_range, data['test_losses'], label=name,
                    color=color, linewidth=2, alpha=0.8)

    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Test Loss', fontsize=11)
    axes[1].set_title('Test Loss Comparison', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('pytorch_optimizer_comparison.png', dpi=100, bbox_inches='tight')
    print("📊 优化器对比已保存: pytorch_optimizer_comparison.png")
    plt.close()


# ==================== 2. 学习率调度 ====================
def demo_lr_schedulers():
    """
    演示 PyTorch 的学习率调度器

    ====================================================================
    🔑 为什么需要学习率调度？
    ====================================================================

    固定学习率问题：
        - 开始：学习率太大 → 震荡
        - 开始：学习率太小 → 太慢
        - 后期：学习率太大 → 无法精细调整

    解决方案：动态调整学习率
        - 开始：大学习率，快速接近
        - 后期：小学习率，精细调整

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("3. 学习率调度器演示")
    print("=" * 70)

    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # ========== StepLR ==========
    print("\n" + "-" * 70)
    print("StepLR - 阶梯衰减")
    print("-" * 70)
    print("""
每隔 step_size 个 epoch，学习率乘以 gamma

例子: step_size=10, gamma=0.5
    epoch 0-9:   lr = 0.1
    epoch 10-19: lr = 0.05
    epoch 20-29: lr = 0.025
    """)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    print(f"创建 StepLR: {scheduler}")

    # ========== ExponentialLR ==========
    print("\n" + "-" * 70)
    print("ExponentialLR - 指数衰减")
    print("-" * 70)
    print("""
每个 epoch，学习率乘以 gamma

lr = lr₀ × gamma^epoch
平滑下降
    """)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # ========== CosineAnnealingLR ==========
    print("\n" + "-" * 70)
    print("CosineAnnealingLR - 余弦退火")
    print("-" * 70)
    print("""
学习率按余弦曲线下降

lr = lr_min + (lr_max - lr_min) × (1 + cos(π×T_cur/T_max)) / 2

特点: 平滑下降，现代训练常用
    """)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    # ========== ReduceLROnPlateau ==========
    print("\n" + "-" * 70)
    print("ReduceLROnPlateau - 自适应降低")
    print("-" * 70)
    print("""
当指标（如验证损失）停止改善时，降低学习率

适用场景：不知道何时降低学习率
自动检测plateau（平台期）
    """)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)

    # ========== CyclicLR ==========
    print("\n" + "-" * 70)
    print("CyclicLR - 循环学习率")
    print("-" * 70)
    print("""
学习率在 base_lr 和 max_lr 之间循环

有助于跳出局部最小值
Leslie Smith 提出（超收敛）
    """)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001,
                                           max_lr=0.1, step_size_up=2000)

    # ========== OneCycleLR ==========
    print("\n" + "-" * 70)
    print("OneCycleLR - 单周期策略")
    print("-" * 70)
    print("""
先增大学习率（warm-up），再减小

1. Warm-up: 0 → max_lr
2. Annealing: max_lr → min_lr

现代训练常用，收敛快
    """)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1,
                                              steps_per_epoch=100, epochs=10)

    # 可视化所有调度器
    visualize_lr_schedulers()


def visualize_lr_schedulers():
    """可视化不同的学习率调度器"""
    print("\n可视化学习率调度策略...")

    model = nn.Linear(1, 1)
    n_epochs = 100
    steps_per_epoch = 10

    schedules = {}

    # StepLR
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    lrs = []
    for epoch in range(n_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        for _ in range(steps_per_epoch):
            optimizer.step()
        scheduler.step()
    schedules['StepLR'] = lrs

    # ExponentialLR
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    lrs = []
    for epoch in range(n_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        for _ in range(steps_per_epoch):
            optimizer.step()
        scheduler.step()
    schedules['ExponentialLR'] = lrs

    # CosineAnnealingLR
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    lrs = []
    for epoch in range(n_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        for _ in range(steps_per_epoch):
            optimizer.step()
        scheduler.step()
    schedules['CosineAnnealingLR'] = lrs

    # OneCycleLR
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=n_epochs)
    lrs = []
    for epoch in range(n_epochs):
        epoch_lrs = []
        for _ in range(steps_per_epoch):
            optimizer.step()
            epoch_lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        lrs.append(np.mean(epoch_lrs))
    schedules['OneCycleLR'] = lrs

    # Constant
    schedules['Constant'] = [0.1] * n_epochs

    # 绘图
    plt.figure(figsize=(12, 7))

    colors = ['gray', 'blue', 'green', 'red', 'purple']
    epochs_range = range(n_epochs)

    for (name, lrs), color in zip(schedules.items(), colors):
        plt.plot(epochs_range, lrs, label=name, linewidth=2.5, alpha=0.8, color=color)

    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Learning Rate', fontsize=11)
    plt.title('PyTorch Learning Rate Schedulers', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('pytorch_lr_schedulers.png', dpi=100, bbox_inches='tight')
    print("📊 学习率调度器已保存: pytorch_lr_schedulers.png")
    plt.close()


# ==================== 3. 正则化技术 ====================
class ModelWithDropout(nn.Module):
    """带 Dropout 的模型"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(ModelWithDropout, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # Dropout
        x = torch.sigmoid(self.fc3(x))
        return x


class ModelWithBatchNorm(nn.Module):
    """带 Batch Normalization 的模型"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModelWithBatchNorm, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch Norm
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Norm
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Batch Norm
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)  # Batch Norm
        x = torch.relu(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def demo_regularization():
    """
    演示正则化技术

    ====================================================================
    🔑 Dropout
    ====================================================================

    作用：
        - 训练时：随机关闭一些神经元（设为0）
        - 测试时：所有神经元工作

    为什么有效：
        - 防止神经元共适应（co-adaptation）
        - 类似集成学习（ensemble）
        - 每个神经元学到更鲁棒的特征

    使用：
        nn.Dropout(p=0.5)  # 50% 的神经元被关闭

    ====================================================================
    🔑 Batch Normalization
    ====================================================================

    作用：
        - 归一化每个 batch 的激活值
        - 均值=0，方差=1

    为什么有效：
        - 稳定训练（减少内部协变量偏移）
        - 允许更大的学习率
        - 自带正则化效果

    使用：
        nn.BatchNorm1d(num_features)  # 全连接层
        nn.BatchNorm2d(num_channels)  # 卷积层

    ====================================================================
    🔑 L2 Regularization (Weight Decay)
    ====================================================================

    作用：
        - 惩罚大的权重
        - Loss = Loss + λ × ||W||²

    为什么有效：
        - 防止权重过大
        - 鼓励简单模型

    使用：
        optimizer = optim.Adam(model.parameters(), weight_decay=0.01)

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("4. 正则化技术演示")
    print("=" * 70)

    # 准备数据（容易过拟合的小数据集）
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = make_moons(n_samples=100, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f"\n小数据集（容易过拟合）:")
    print(f"  训练集: {len(X_train)} samples")
    print(f"  测试集: {len(X_test)} samples")

    # 训练不同模型
    models_config = {
        'No Regularization': nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        ),
        'With Dropout': ModelWithDropout(2, 32, 1, dropout_rate=0.5),
        'With BatchNorm': ModelWithBatchNorm(2, 32, 1),
    }

    criterion = nn.BCELoss()
    n_epochs = 100

    results = {}

    for name, model in models_config.items():
        print(f"\n训练 {name}...")

        # L2 正则化 (weight_decay)
        if name == 'No Regularization':
            optimizer = optim.Adam(model.parameters(), lr=0.01)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

        train_losses = []
        test_losses = []

        for epoch in range(n_epochs):
            # 训练
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(train_loader))

            # 测试
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_loss = criterion(test_output, y_test)
                test_losses.append(test_loss.item())

        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses
        }

        print(f"  最终训练损失: {train_losses[-1]:.4f}")
        print(f"  最终测试损失: {test_losses[-1]:.4f}")
        print(f"  过拟合差距: {abs(train_losses[-1] - test_losses[-1]):.4f}")

    # 可视化
    visualize_regularization(results, n_epochs)


def visualize_regularization(results, n_epochs):
    """可视化正则化效果"""
    print("\n可视化正则化效果...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['red', 'blue', 'green']
    epochs_range = range(1, n_epochs + 1)

    # Plot 1: 训练损失
    for (name, data), color in zip(results.items(), colors):
        axes[0].plot(epochs_range, data['train_losses'], label=name,
                    color=color, linewidth=2, alpha=0.8)

    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Training Loss', fontsize=11)
    axes[0].set_title('Training Loss (Lower = Better)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Plot 2: 测试损失
    for (name, data), color in zip(results.items(), colors):
        axes[1].plot(epochs_range, data['test_losses'], label=name,
                    color=color, linewidth=2, alpha=0.8)

    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Test Loss', fontsize=11)
    axes[1].set_title('Test Loss (Lower = Better, Shows Overfitting)',
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('pytorch_regularization.png', dpi=100, bbox_inches='tight')
    print("📊 正则化效果已保存: pytorch_regularization.png")
    plt.close()


# ==================== 4. 梯度裁剪 ====================
def demo_gradient_clipping():
    """
    演示梯度裁剪

    ====================================================================
    🔑 为什么需要梯度裁剪？
    ====================================================================

    问题：梯度爆炸
        - 梯度太大 → 参数更新太大 → 发散
        - 常见于：RNN、LSTM、深层网络

    解决：梯度裁剪
        - 限制梯度的最大值
        - 保持梯度方向，只缩放大小

    ====================================================================
    🔑 两种方式
    ====================================================================

    1. Clip by Value（按值裁剪）
       gradient = max(min(gradient, max_value), -max_value)

    2. Clip by Norm（按范数裁剪）
       if ||gradient|| > max_norm:
           gradient = gradient × (max_norm / ||gradient||)

    PyTorch 实现（推荐 Clip by Norm）：
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("5. 梯度裁剪演示")
    print("=" * 70)

    print("""
梯度裁剪在训练循环中使用：

for epoch in range(n_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), target)
        loss.backward()

        # 梯度裁剪（在 optimizer.step() 之前）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

何时使用：
    - RNN/LSTM：防止梯度爆炸（max_norm=1.0）
    - Transformer：稳定训练（max_norm=1.0）
    - 深层网络：发现梯度爆炸时
    - GAN：稳定训练

常用值：
    - max_norm = 1.0（RNN/LSTM）
    - max_norm = 5.0（Transformer）
    - max_norm = 10.0（GAN）
    """)


# ==================== 5. 权重初始化 ====================
def demo_weight_initialization():
    """
    演示权重初始化

    ====================================================================
    🔑 为什么权重初始化重要？
    ====================================================================

    糟糕的初始化：
        - 全0：所有神经元学到同样的东西
        - 太大：激活值爆炸
        - 太小：激活值消失

    好的初始化：
        - 打破对称性
        - 保持激活值方差
        - 加速收敛

    ====================================================================
    🔑 常用初始化方法
    ====================================================================

    1. Xavier/Glorot Initialization
       - 用于: Sigmoid, Tanh
       - 公式: Uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))

    2. He Initialization
       - 用于: ReLU
       - 公式: Normal(0, √(2/fan_in))

    3. Orthogonal Initialization
       - 用于: RNN, LSTM
       - 创建正交矩阵

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("6. 权重初始化演示")
    print("=" * 70)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    # ========== Xavier/Glorot Initialization ==========
    print("\n" + "-" * 70)
    print("Xavier/Glorot Initialization (for Sigmoid/Tanh)")
    print("-" * 70)

    def init_xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_xavier)
    print("应用 Xavier 初始化")

    # ========== He Initialization ==========
    print("\n" + "-" * 70)
    print("He Initialization (for ReLU)")
    print("-" * 70)

    def init_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_he)
    print("应用 He 初始化（推荐用于ReLU）")

    # ========== Orthogonal Initialization ==========
    print("\n" + "-" * 70)
    print("Orthogonal Initialization (for RNN/LSTM)")
    print("-" * 70)

    def init_orthogonal(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_orthogonal)
    print("应用 Orthogonal 初始化")

    print("""
使用建议：
    - ReLU 激活函数 → He Initialization（默认）
    - Sigmoid/Tanh → Xavier Initialization
    - RNN/LSTM → Orthogonal Initialization
    - PyTorch 默认使用 Kaiming (He) 初始化
    """)


# ==================== 6. 完整训练示例 ====================
def complete_training_example():
    """
    完整的训练示例（整合所有技术）

    展示如何在实际训练中组合使用：
    - 优化器 (Adam)
    - 学习率调度 (OneCycleLR)
    - 正则化 (Dropout + BatchNorm + Weight Decay)
    - 梯度裁剪
    - 权重初始化
    """
    print("\n" + "=" * 70)
    print("7. 完整训练示例（Best Practices）")
    print("=" * 70)

    # 数据
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 模型（使用 BatchNorm + Dropout）
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    # 权重初始化（He）
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # 优化器（Adam + Weight Decay）
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    # 学习率调度（OneCycleLR）
    n_epochs = 50
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs
    )

    # 损失函数
    criterion = nn.BCELoss()

    print(f"\n配置:")
    print(f"  模型: 3-layer MLP with BatchNorm + Dropout")
    print(f"  优化器: AdamW (weight_decay=0.01)")
    print(f"  学习率调度: OneCycleLR")
    print(f"  正则化: BatchNorm + Dropout(0.3) + Weight Decay")
    print(f"  梯度裁剪: max_norm=1.0")
    print(f"  权重初始化: He (Kaiming)")

    print(f"\n开始训练...")

    train_losses = []
    test_losses = []
    lrs = []

    for epoch in range(n_epochs):
        # 训练
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # OneCycleLR 每个 step 更新

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        lrs.append(optimizer.param_groups[0]['lr'])

        # 测试
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = criterion(test_output, y_test)
            test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/{n_epochs} | "
                  f"Train Loss: {train_losses[-1]:.4f} | "
                  f"Test Loss: {test_losses[-1]:.4f} | "
                  f"LR: {lrs[-1]:.6f}")

    print(f"\n训练完成！")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  最终测试损失: {test_losses[-1]:.4f}")

    # 可视化
    visualize_complete_training(train_losses, test_losses, lrs, n_epochs)


def visualize_complete_training(train_losses, test_losses, lrs, n_epochs):
    """可视化完整训练过程"""
    print("\n可视化训练过程...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    epochs_range = range(1, n_epochs + 1)

    # Plot 1: 损失曲线
    axes[0].plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Progress', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot 2: 学习率变化
    axes[1].plot(epochs_range, lrs, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Learning Rate', fontsize=11)
    axes[1].set_title('Learning Rate Schedule (OneCycleLR)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('pytorch_complete_training.png', dpi=100, bbox_inches='tight')
    print("📊 完整训练过程已保存: pytorch_complete_training.png")
    plt.close()


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("优化与正则化技术 - PyTorch 实现")
    print("=" * 70)

    # 1. 优化器基础
    demo_pytorch_optimizers()

    # 2. 优化器对比
    compare_optimizers_pytorch()

    # 3. 学习率调度
    demo_lr_schedulers()

    # 4. 正则化
    demo_regularization()

    # 5. 梯度裁剪
    demo_gradient_clipping()

    # 6. 权重初始化
    demo_weight_initialization()

    # 7. 完整示例
    complete_training_example()

    # 8. 总结
    print("\n" + "=" * 70)
    print("✅ 核心要点总结")
    print("=" * 70)
    print("""
1. PyTorch 优化器

   常用优化器：
   - optim.SGD(params, lr, momentum)
   - optim.Adam(params, lr, betas)
   - optim.AdamW(params, lr, weight_decay)  ← 推荐

   使用：
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

2. 学习率调度

   常用调度器：
   - StepLR: 阶梯衰减
   - CosineAnnealingLR: 余弦退火
   - OneCycleLR: 单周期（推荐）
   - ReduceLROnPlateau: 自适应

   使用：
   scheduler = optim.lr_scheduler.OneCycleLR(optimizer, ...)
   scheduler.step()  # 每个 epoch 或 batch 后调用

3. 正则化技术

   Dropout:
   nn.Dropout(p=0.5)  # 训练时关闭50%神经元

   Batch Normalization:
   nn.BatchNorm1d(num_features)  # 全连接层
   nn.BatchNorm2d(num_channels)  # 卷积层

   Weight Decay (L2):
   optimizer = optim.Adam(params, weight_decay=0.01)

4. 梯度裁剪

   防止梯度爆炸：
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

   使用场景：
   - RNN/LSTM: max_norm=1.0
   - Transformer: max_norm=1.0-5.0

5. 权重初始化

   He (Kaiming) - for ReLU:
   nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

   Xavier/Glorot - for Sigmoid/Tanh:
   nn.init.xavier_uniform_(m.weight)

   Orthogonal - for RNN:
   nn.init.orthogonal_(m.weight)

6. 完整训练流程（Best Practices）

   # 1. 模型（BatchNorm + Dropout）
   model = nn.Sequential(
       nn.Linear(input_dim, hidden_dim),
       nn.BatchNorm1d(hidden_dim),
       nn.ReLU(),
       nn.Dropout(0.3),
       # ...
   )

   # 2. 权重初始化
   model.apply(init_weights)

   # 3. 优化器 + Weight Decay
   optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

   # 4. 学习率调度
   scheduler = optim.lr_scheduler.OneCycleLR(optimizer, ...)

   # 5. 训练循环
   for epoch in range(n_epochs):
       for batch in dataloader:
           optimizer.zero_grad()
           loss = criterion(model(batch), target)
           loss.backward()

           # 梯度裁剪
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

           optimizer.step()
           scheduler.step()

7. 选择建议

   快速原型：
   - Optimizer: Adam (默认参数)
   - Scheduler: None 或 ReduceLROnPlateau
   - Regularization: Dropout(0.3)

   最佳性能：
   - Optimizer: AdamW (weight_decay=0.01)
   - Scheduler: OneCycleLR 或 CosineAnnealingLR
   - Regularization: BatchNorm + Dropout + Weight Decay

   RNN/LSTM：
   - Optimizer: Adam
   - Gradient Clipping: max_norm=1.0
   - Init: Orthogonal

   Transformer：
   - Optimizer: AdamW
   - Scheduler: OneCycleLR with Warmup
   - Gradient Clipping: max_norm=1.0

8. 调参技巧

   学习率：
   - 太大 → 发散、震荡
   - 太小 → 收敛慢
   - 常用范围: 0.0001 ~ 0.01
   - 使用 Learning Rate Finder

   Dropout Rate：
   - 全连接层: 0.3-0.5
   - 卷积层: 0.1-0.3
   - RNN: 0.1-0.2

   Weight Decay：
   - 轻度: 0.0001
   - 中度: 0.001-0.01
   - 重度: 0.1

9. 调试技巧

   过拟合（Train好，Test差）：
   → 增加 Dropout
   → 增加 Weight Decay
   → 减少模型复杂度
   → 增加数据

   欠拟合（Train Test都差）：
   → 增加模型复杂度
   → 减少正则化
   → 训练更久

   梯度爆炸：
   → 添加 Gradient Clipping
   → 降低学习率
   → 使用 BatchNorm

   训练不稳定：
   → 使用 BatchNorm
   → 降低学习率
   → 使用 OneCycleLR with Warmup

10. PyTorch vs NumPy

    NumPy:
    - 手写优化器（理解原理）
    - 代码量大
    - 速度慢

    PyTorch:
    - 内置优化器（工业实践）
    - 代码简洁
    - GPU 加速
    - 自动微分
    """)


if __name__ == "__main__":
    main()

    print("\n💡 练习建议:")
    print("  1. 对比不同优化器在自己数据上的表现")
    print("  2. 实验不同的学习率调度策略")
    print("  3. 调整 Dropout rate，观察过拟合变化")
    print("  4. 理解 BatchNorm 如何稳定训练")
    print("  5. 实现完整训练流程（整合所有技术）")
    print("  6. 使用 tensorboard 可视化训练过程")
