"""
卷积神经网络 (CNN) - PyTorch 实现

对比 NumPy 版本：
- NumPy: 手写卷积、池化，理解数学原理
- PyTorch: 使用框架，GPU加速，工业实践

本文件内容：
1. PyTorch CNN 基础组件
2. 完整的 CNN 模型（MNIST 数字识别）
3. GPU 训练加速
4. 训练可视化
5. 与 NumPy 版本性能对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time


# ==================== 1. PyTorch CNN 基础组件 ====================
def demo_pytorch_conv():
    """
    演示 PyTorch 的卷积操作

    ====================================================================
    🔑 PyTorch vs NumPy 卷积
    ====================================================================

    NumPy 版本（手写）：
    ```python
    def conv2d(image, kernel):
        for i in range(out_h):
            for j in range(out_w):
                window = image[i:i+k, j:j+k]
                output[i, j] = np.sum(window * kernel)
    ```

    PyTorch 版本（一行）：
    ```python
    output = F.conv2d(image, kernel)
    ```

    PyTorch 帮你做了什么？
    - 自动批量处理（batch）
    - 自动GPU加速
    - 自动计算梯度（反向传播）
    - 数值优化（更快更稳定）

    ====================================================================
    """
    print("=" * 70)
    print("1. PyTorch 卷积操作演示")
    print("=" * 70)

    # 创建输入图像（batch_size=1, channels=1, height=5, width=5）
    # PyTorch 格式：(N, C, H, W)
    image = torch.tensor([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度

    print(f"\n输入图像 shape: {image.shape}")  # (1, 1, 5, 5)

    # 创建卷积核（垂直边缘检测）
    # PyTorch 格式：(out_channels, in_channels, height, width)
    kernel = torch.tensor([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

    print(f"卷积核 shape: {kernel.shape}")  # (1, 1, 3, 3)

    # 执行卷积（超级简单！）
    output = F.conv2d(image, kernel)

    print(f"输出 shape: {output.shape}")  # (1, 1, 3, 3)
    print(f"\n卷积输出:\n{output.squeeze()}")

    # 使用 nn.Conv2d（推荐方式，可学习参数）
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

    # 手动设置权重（和上面的kernel一样）
    with torch.no_grad():
        conv_layer.weight = nn.Parameter(kernel)
        conv_layer.bias = nn.Parameter(torch.zeros(1))

    output2 = conv_layer(image)
    print(f"\nnn.Conv2d 输出:\n{output2.squeeze()}")

    print("\n💡 PyTorch 优势:")
    print("  - 一行代码完成卷积")
    print("  - 自动支持 batch 处理")
    print("  - 自动计算梯度（反向传播）")
    print("  - GPU 加速（添加 .cuda()）")


def demo_pytorch_pooling():
    """演示 PyTorch 的池化操作"""
    print("\n" + "=" * 70)
    print("2. PyTorch 池化操作演示")
    print("=" * 70)

    # 创建输入
    x = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    print(f"\n输入 shape: {x.shape}")  # (1, 1, 4, 4)
    print(f"输入:\n{x.squeeze()}")

    # MaxPooling (2×2)
    pooled = F.max_pool2d(x, kernel_size=2)
    print(f"\nMaxPool2d 输出 shape: {pooled.shape}")  # (1, 1, 2, 2)
    print(f"输出:\n{pooled.squeeze()}")

    # 使用 nn.MaxPool2d
    pool_layer = nn.MaxPool2d(kernel_size=2)
    pooled2 = pool_layer(x)
    print(f"\nnn.MaxPool2d 输出:\n{pooled2.squeeze()}")


# ==================== 2. 完整的 CNN 模型 ====================
class SimpleCNN(nn.Module):
    """
    简单的 CNN 模型（MNIST 数字识别）

    ====================================================================
    🔑 PyTorch 模型定义
    ====================================================================

    PyTorch 定义模型有两步：
    1. __init__: 定义层（layer）
    2. forward: 定义前向传播逻辑

    对比 NumPy：
    - NumPy: 手动管理所有权重，手写前向传播
    - PyTorch: 定义层结构，自动管理权重，自动反向传播

    ====================================================================

    网络结构：
    Input (1, 28, 28)
        ↓
    Conv1 (32, 26, 26)  # 3×3 卷积，32个卷积核
        ↓ ReLU
    Pool1 (32, 13, 13)  # 2×2 最大池化
        ↓
    Conv2 (64, 11, 11)  # 3×3 卷积，64个卷积核
        ↓ ReLU
    Pool2 (64, 5, 5)    # 2×2 最大池化
        ↓
    Flatten (1600)      # 展平
        ↓
    FC1 (128)           # 全连接层
        ↓ ReLU + Dropout
    FC2 (10)            # 输出层（10个类别）
        ↓ Softmax (隐式在loss中)
    Output (10)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积层1: 1 → 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        # 卷积层2: 32 → 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # 池化层（2×2）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层1: 64*5*5 → 128
        # 为什么是 5×5？
        # 28 → (卷积-2=26) → (池化/2=13) → (卷积-2=11) → (池化/2=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 全连接层2: 128 → 10 (10个数字类别)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        前向传播

        x: (batch_size, 1, 28, 28)
        返回: (batch_size, 10)
        """
        # 卷积层1 + ReLU + 池化
        x = self.conv1(x)           # (batch, 32, 26, 26)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 32, 13, 13)

        # 卷积层2 + ReLU + 池化
        x = self.conv2(x)           # (batch, 64, 11, 11)
        x = F.relu(x)
        x = self.pool(x)            # (batch, 64, 5, 5)

        # 展平
        x = x.view(-1, 64 * 5 * 5)  # (batch, 1600)
        # 也可以用: x = torch.flatten(x, 1)

        # 全连接层1 + ReLU + Dropout
        x = self.fc1(x)             # (batch, 128)
        x = F.relu(x)
        x = self.dropout(x)

        # 全连接层2（输出层）
        x = self.fc2(x)             # (batch, 10)

        return x


# ==================== 3. 训练和评估 ====================
def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """训练一个 epoch"""
    model.train()  # 设置为训练模式（启用 Dropout）

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 移到 GPU/CPU
        data, target = data.to(device), target.to(device)

        # 清零梯度（重要！）
        optimizer.zero_grad()

        # 前向传播
        output = model(data)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播（自动计算梯度！）
        loss.backward()

        # 更新权重
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # 打印进度
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """评估模型"""
    model.eval()  # 设置为评估模式（禁用 Dropout）

    test_loss = 0
    correct = 0

    with torch.no_grad():  # 不计算梯度（节省内存和时间）
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 前向传播
            output = model(data)

            # 累计损失
            test_loss += criterion(output, target).item()

            # 预测
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy


def train_cnn():
    """
    完整训练流程

    ====================================================================
    🔑 PyTorch 训练流程
    ====================================================================

    1. 准备数据
       - 使用 DataLoader（自动批处理、打乱、多线程）

    2. 定义模型
       - 继承 nn.Module
       - 定义 __init__ 和 forward

    3. 定义损失函数和优化器
       - 损失函数：CrossEntropyLoss（分类）、MSELoss（回归）
       - 优化器：Adam、SGD、RMSprop

    4. 训练循环
       for epoch in range(n_epochs):
           for batch in train_loader:
               optimizer.zero_grad()    # 清零梯度
               output = model(batch)    # 前向传播
               loss = criterion(output, target)  # 计算损失
               loss.backward()          # 反向传播
               optimizer.step()         # 更新权重

    5. 评估
       - model.eval() + torch.no_grad()

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("3. 训练 CNN 模型（MNIST）")
    print("=" * 70)

    # ========== 1. 检查 GPU ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    if torch.cuda.is_available():
        print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  ⚠️ 没有检测到 GPU，使用 CPU 训练（会慢很多）")

    # ========== 2. 准备数据 ==========
    print("\n准备数据...")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 Tensor，范围 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化（MNIST 的均值和标准差）
    ])

    # 下载并加载 MNIST 数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # DataLoader（自动批处理 + 打乱）
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  测试集: {len(test_dataset)} 张图片")
    print(f"  Batch size: {batch_size}")

    # ========== 3. 创建模型 ==========
    print("\n创建模型...")
    model = SimpleCNN().to(device)  # 移到 GPU/CPU

    # 打印模型结构
    print(model)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # ========== 4. 定义损失函数和优化器 ==========
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务）
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

    # ========== 5. 训练 ==========
    print("\n开始训练...")
    n_epochs = 5

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        print("-" * 70)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )

        # 评估
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        # 记录
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 打印结果
        print(f"\n  训练集 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  测试集 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.2f} 秒")

    # ========== 6. 可视化 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    epochs_range = range(1, n_epochs + 1)
    axes[0].plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2)
    axes[0].plot(epochs_range, test_losses, 'r-o', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training and Test Loss', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs_range, train_accs, 'b-o', label='Training Acc', linewidth=2)
    axes[1].plot(epochs_range, test_accs, 'r-o', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Training and Test Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('cnn_pytorch_training.png', dpi=100, bbox_inches='tight')
    print("\n📊 训练曲线已保存: cnn_pytorch_training.png")
    plt.close()

    # ========== 7. 可视化预测结果 ==========
    visualize_predictions(model, device, test_loader)

    return model, test_acc


def visualize_predictions(model, device, test_loader):
    """可视化模型预测"""
    print("\n可视化预测结果...")

    model.eval()

    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # 预测
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    # 可视化前16张图片
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(16):
        img = images[i].cpu().numpy().squeeze()
        true_label = labels[i].item()
        pred_label = predictions[i].item()

        axes[i].imshow(img, cmap='gray')

        # 标题颜色：预测正确=绿色，错误=红色
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}',
                         color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('cnn_pytorch_predictions.png', dpi=100, bbox_inches='tight')
    print("📊 预测结果已保存: cnn_pytorch_predictions.png")
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
      - 理解数学原理
      - 从零实现，掌握细节
      - 不依赖深度学习框架

    ❌ 缺点：
      - 代码量大（需要手写反向传播）
      - 速度慢（无GPU加速）
      - 难以扩展（复杂模型难实现）
      - 数值不稳定（需要手动处理）

    PyTorch 版本：
    ✅ 优点：
      - 代码简洁（几行搞定）
      - GPU 加速（快100倍以上）
      - 自动微分（不需要手写反向传播）
      - 稳定（数值优化做得好）
      - 工业界标准（实际工作中使用）

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
性能对比（MNIST 数字识别）：

+----------------+------------------+------------------+
|     指标       |   NumPy 版本     |  PyTorch 版本    |
+----------------+------------------+------------------+
| 代码量         | ~500 行          | ~100 行          |
| 训练时间       | ~10 分钟 (CPU)   | ~30 秒 (GPU)     |
| 测试准确率     | ~90%             | ~98%             |
| GPU 支持       | ❌               | ✅               |
| 自动微分       | ❌ (手写)        | ✅               |
| 可扩展性       | ❌               | ✅               |
| 工业应用       | ❌               | ✅               |
+----------------+------------------+------------------+

代码对比：

NumPy 版本（复杂）：
```python
# 需要手写前向传播
def forward(self, X):
    self.z1 = np.dot(X, self.W1) + self.b1
    self.a1 = relu(self.z1)
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    self.a2 = softmax(self.z2)
    return self.a2

# 需要手写反向传播
def backward(self, X, y):
    m = X.shape[0]
    dz2 = self.a2 - y
    dW2 = (1/m) * np.dot(self.a1.T, dz2)
    db2 = (1/m) * np.sum(dz2, axis=0)
    da1 = np.dot(dz2, self.W2.T)
    dz1 = da1 * relu_derivative(self.z1)
    dW1 = (1/m) * np.dot(X.T, dz1)
    db1 = (1/m) * np.sum(dz1, axis=0)
    # ... 更新权重
```

PyTorch 版本（简洁）：
```python
# 定义模型
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练（自动微分！）
output = model(data)
loss = criterion(output, target)
loss.backward()          # ← 自动计算梯度！
optimizer.step()         # ← 自动更新权重！
```

总结：
- 学习原理 → 用 NumPy
- 实际应用 → 用 PyTorch
- 两者结合 → 最佳理解！
    """)


# ==================== 5. 主程序 ====================
def main():
    print("=" * 70)
    print("卷积神经网络 (CNN) - PyTorch 实现")
    print("=" * 70)

    # 1. PyTorch 基础组件
    demo_pytorch_conv()
    demo_pytorch_pooling()

    # 2. 训练完整模型
    model, test_acc = train_cnn()

    # 3. 对比 PyTorch vs NumPy
    compare_pytorch_vs_numpy()

    # 4. 总结
    print("\n" + "=" * 70)
    print("✅ 核心要点总结")
    print("=" * 70)
    print("""
1. PyTorch CNN 基础组件

   卷积层：
   nn.Conv2d(in_channels, out_channels, kernel_size)

   池化层：
   nn.MaxPool2d(kernel_size)

   全连接层：
   nn.Linear(in_features, out_features)

2. 定义模型（继承 nn.Module）

   class MyCNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(...)
           self.fc1 = nn.Linear(...)

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           return x

3. 训练流程

   # 准备
   model = MyCNN().to(device)  # GPU 加速
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters())

   # 训练循环
   for epoch in range(n_epochs):
       for batch in train_loader:
           optimizer.zero_grad()     # 清零梯度
           output = model(batch)     # 前向传播
           loss = criterion(output, target)  # 计算损失
           loss.backward()           # 反向传播（自动！）
           optimizer.step()          # 更新权重

4. GPU 加速

   # 检查 GPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # 移动模型和数据到 GPU
   model = model.to(device)
   data = data.to(device)

   # 速度提升：CPU 10分钟 → GPU 30秒（20倍）

5. PyTorch vs NumPy

   NumPy：
   - 理解原理（手写卷积、反向传播）
   - 代码量大
   - 速度慢

   PyTorch：
   - 工业实践（GPU、自动微分）
   - 代码简洁
   - 速度快100倍+

6. 实践建议

   学习路径：
   1. 先看 NumPy 版本（理解数学）
   2. 再看 PyTorch 版本（学习框架）
   3. 对比两个版本（理解框架做了什么）

   实际工作：
   - 100% 用 PyTorch（或 TensorFlow）
   - NumPy 只用于理解原理
    """)


if __name__ == "__main__":
    main()

    print("\n💡 练习建议:")
    print("  1. 修改网络结构（添加更多卷积层）")
    print("  2. 在 CIFAR-10 数据集上训练（彩色图像）")
    print("  3. 使用预训练模型（ResNet、VGG）")
    print("  4. 实现数据增强（transforms）")
    print("  5. 对比不同优化器（SGD、Adam、RMSprop）")
    print("  6. 可视化卷积核学到的特征")
