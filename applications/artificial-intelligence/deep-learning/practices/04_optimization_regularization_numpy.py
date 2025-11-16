"""
Optimization and Regularization Techniques

Problem: How to train deep networks effectively and prevent overfitting?
Goal: Master the essential techniques for training robust neural networks

Core Concepts:
1. Optimizers: How to update weights (SGD, Momentum, Adam)
2. Learning Rate Scheduling: Adaptive learning rates
3. Regularization: Prevent overfitting (Dropout, BatchNorm, L2)
4. Gradient Clipping: Prevent exploding gradients
5. Weight Initialization: Start with good weights

Why These Matter?
- Bad optimizer → Slow convergence or stuck in local minima
- Wrong learning rate → Divergence or too slow
- No regularization → Overfitting (good on training, bad on test)
- Poor initialization → Dead neurons or exploding activations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. Optimizers ====================
class SGD:
    """
    Stochastic Gradient Descent (最基础的优化器)

    ====================================================================
    🔑 What is SGD?
    ====================================================================

    SGD = 随机梯度下降，最简单的优化方法

    更新规则：
        θ = θ - learning_rate × gradient

    例子：
        权重 = 0.5
        梯度 = 0.3（表示往正方向走会增大损失）
        学习率 = 0.1

        新权重 = 0.5 - 0.1 × 0.3 = 0.47

    ====================================================================
    🔑 SGD 的问题
    ====================================================================

    问题1：学习率固定
        - 太大：震荡，不收敛
        - 太小：收敛太慢

    问题2：所有参数用同一个学习率
        - 有些参数需要大步走
        - 有些参数需要小步走

    问题3：容易卡在鞍点（saddle point）
        - 梯度接近0的平坦区域
        - 无法快速逃离

    ====================================================================
    """

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        """
        更新参数

        params: 参数列表 [W1, b1, W2, b2, ...]
        grads: 梯度列表 [dW1, db1, dW2, db2, ...]
        """
        updated_params = []
        for param, grad in zip(params, grads):
            # 简单的梯度下降
            param = param - self.lr * grad
            updated_params.append(param)
        return updated_params


class Momentum:
    """
    Momentum 优化器（带动量）

    ====================================================================
    🔑 What is Momentum?
    ====================================================================

    Momentum = 动量，像滚雪球一样累积梯度

    核心思想：
        - 不只看当前梯度
        - 还要考虑之前的"惯性"

    更新规则：
        velocity = momentum × velocity - learning_rate × gradient
        θ = θ + velocity

    类比：滚球下山
        - 不是每次都按当前坡度走
        - 而是积累了速度，有惯性
        - 可以冲过小坑（局部最小值）

    ====================================================================
    🔑 Momentum vs SGD
    ====================================================================

    SGD：
        像走路，每步都重新看方向
        遇到小坑就停下了

    Momentum：
        像滚球，有惯性
        可以冲过小坑
        在一致的方向上加速

    参数 β（momentum 系数）：
        - β = 0: 退化为 SGD
        - β = 0.9: 常用值，保留90%的旧速度
        - β = 0.99: 更强的惯性

    ====================================================================
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            # 第一次，初始化速度为0
            self.velocity = [np.zeros_like(p) for p in params]

        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # 更新速度：旧速度 × momentum - 学习率 × 梯度
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad

            # 更新参数
            param = param + self.velocity[i]
            updated_params.append(param)

        return updated_params


class RMSprop:
    """
    RMSprop 优化器（均方根传播）

    ====================================================================
    🔑 What is RMSprop?
    ====================================================================

    RMSprop = Root Mean Square Propagation

    核心思想：
        - 自适应学习率
        - 对频繁变化的参数用小学习率
        - 对稀疏变化的参数用大学习率

    更新规则：
        cache = decay × cache + (1-decay) × gradient²
        θ = θ - learning_rate × gradient / (√cache + ε)

    解释：
        - cache：累积的梯度平方（代表梯度的"历史大小"）
        - 梯度大的参数 → cache大 → 除以大数 → 步长变小
        - 梯度小的参数 → cache小 → 除以小数 → 步长变大

    ====================================================================
    🔑 为什么有效？
    ====================================================================

    问题场景：
        参数1：梯度一直很大（震荡）
        参数2：梯度一直很小（慢）

    SGD：
        参数1 → 步长大 → 震荡不收敛
        参数2 → 步长小 → 收敛太慢

    RMSprop：
        参数1 → cache大 → 自动减小步长 → 平稳
        参数2 → cache小 → 自动增大步长 → 加速

    ====================================================================
    """

    def __init__(self, learning_rate=0.01, decay=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None

    def update(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]

        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # 累积梯度平方
            self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * grad**2

            # 自适应学习率更新
            param = param - self.lr * grad / (np.sqrt(self.cache[i]) + self.epsilon)
            updated_params.append(param)

        return updated_params


class Adam:
    """
    Adam 优化器（Adaptive Moment Estimation）

    ====================================================================
    🔑 What is Adam?
    ====================================================================

    Adam = Momentum + RMSprop 的结合体（最流行的优化器！）

    结合了两个优点：
        1. Momentum：保留梯度的方向（一阶矩）
        2. RMSprop：自适应学习率（二阶矩）

    更新规则：
        # 一阶矩（动量）
        m = β₁ × m + (1-β₁) × gradient

        # 二阶矩（自适应学习率）
        v = β₂ × v + (1-β₂) × gradient²

        # 偏差修正（bias correction）
        m_hat = m / (1 - β₁ᵗ)
        v_hat = v / (1 - β₂ᵗ)

        # 更新参数
        θ = θ - learning_rate × m_hat / (√v_hat + ε)

    ====================================================================
    🔑 为什么 Adam 这么好？
    ====================================================================

    1. 自适应学习率（来自 RMSprop）
       - 不同参数自动调整步长
       - 不需要手动调学习率

    2. 动量加速（来自 Momentum）
       - 加速收敛
       - 可以冲过小坑

    3. 偏差修正
       - 开始时 m 和 v 接近0（初始化）
       - 修正后更准确

    4. 鲁棒性强
       - 默认参数 (β₁=0.9, β₂=0.999) 通常就很好
       - 适用于大多数问题

    ====================================================================
    🔑 超参数选择
    ====================================================================

    learning_rate (α):
        - 默认：0.001
        - 范围：0.0001 ~ 0.01

    β₁ (momentum):
        - 默认：0.9
        - 一般不需要改

    β₂ (RMSprop decay):
        - 默认：0.999
        - 一般不需要改

    ε (数值稳定性):
        - 默认：1e-8
        - 防止除以0

    ====================================================================
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩（动量）
        self.v = None  # 二阶矩（自适应学习率）
        self.t = 0     # 时间步

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1  # 时间步 +1

        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            # 更新一阶矩（动量）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 更新二阶矩（自适应学习率）
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # 更新参数
            param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(param)

        return updated_params


def compare_optimizers():
    """比较不同优化器的收敛速度"""
    print("=" * 70)
    print("Comparison: Different Optimizers")
    print("=" * 70)

    # 生成非凸优化问题（Beale函数）
    def beale_function(x, y):
        """Beale函数（经典的优化测试函数）"""
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        return term1 + term2 + term3

    def beale_gradient(x, y):
        """Beale函数的梯度"""
        term1 = 1.5 - x + x*y
        term2 = 2.25 - x + x*y**2
        term3 = 2.625 - x + x*y**3

        dx = 2*term1*(-1+y) + 2*term2*(-1+y**2) + 2*term3*(-1+y**3)
        dy = 2*term1*x + 2*term2*2*x*y + 2*term3*3*x*y**2

        return np.array([dx, dy])

    # 初始点
    start_point = np.array([3.0, 3.0])

    # 不同优化器
    optimizers = {
        'SGD': SGD(learning_rate=0.001),
        'Momentum': Momentum(learning_rate=0.001, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.01, decay=0.9),
        'Adam': Adam(learning_rate=0.01),
    }

    # 训练
    n_iterations = 200
    trajectories = {}
    losses = {}

    for name, optimizer in optimizers.items():
        print(f"\nOptimizing with {name}...")

        point = start_point.copy()
        trajectory = [point.copy()]
        loss_history = [beale_function(point[0], point[1])]

        for i in range(n_iterations):
            # 计算梯度
            grad = beale_gradient(point[0], point[1])

            # 更新参数
            updated = optimizer.update([point], [grad])
            point = updated[0]

            # 记录
            trajectory.append(point.copy())
            loss_history.append(beale_function(point[0], point[1]))

        trajectories[name] = np.array(trajectory)
        losses[name] = loss_history

        print(f"  Final point: ({point[0]:.4f}, {point[1]:.4f})")
        print(f"  Final loss: {loss_history[-1]:.6f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: 损失曲线
    for name, loss_history in losses.items():
        axes[0].plot(loss_history, label=name, linewidth=2, alpha=0.8)

    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Convergence Speed Comparison', fontsize=12, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot 2: 优化轨迹
    x = np.linspace(-0.5, 4.5, 100)
    y = np.linspace(-0.5, 4.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = beale_function(X, Y)

    axes[1].contour(X, Y, Z, levels=np.logspace(-1, 3, 20), alpha=0.3)

    colors = {'SGD': 'red', 'Momentum': 'blue', 'RMSprop': 'green', 'Adam': 'purple'}
    for name, trajectory in trajectories.items():
        axes[1].plot(trajectory[:, 0], trajectory[:, 1],
                    '-o', color=colors[name], label=name,
                    markersize=3, linewidth=1.5, alpha=0.7)
        # 标记起点和终点
        axes[1].plot(trajectory[0, 0], trajectory[0, 1], 'ko', markersize=10)
        axes[1].plot(trajectory[-1, 0], trajectory[-1, 1], 'k*', markersize=15)

    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('y', fontsize=11)
    axes[1].set_title('Optimization Trajectories\n(Black circle=start, star=end)',
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=100, bbox_inches='tight')
    print("\n📊 Comparison saved to: optimizer_comparison.png")
    plt.close()

    print("\n💡 Observations:")
    print("  - SGD: Slowest, may get stuck")
    print("  - Momentum: Faster, can overshoot")
    print("  - RMSprop: Adaptive, smoother")
    print("  - Adam: Best of both worlds, usually fastest")


# ==================== 2. Learning Rate Scheduling ====================
class LearningRateScheduler:
    """
    学习率调度器

    ====================================================================
    🔑 为什么需要学习率调度？
    ====================================================================

    固定学习率的问题：
        - 开始时：学习率太小 → 收敛慢
        - 开始时：学习率太大 → 震荡
        - 后期：学习率太大 → 无法精细调整

    解决方案：动态调整学习率
        - 开始：大学习率，快速接近最优点
        - 后期：小学习率，精细调整

    ====================================================================
    🔑 常见策略
    ====================================================================

    1. Step Decay（阶梯衰减）
       lr = lr₀ × 0.5^(epoch / step_size)

       例子：每10个epoch减半
       epoch 0-9:   lr = 0.1
       epoch 10-19: lr = 0.05
       epoch 20-29: lr = 0.025

    2. Exponential Decay（指数衰减）
       lr = lr₀ × e^(-k×epoch)

       平滑下降，没有突变

    3. Cosine Annealing（余弦退火）
       lr = lr_min + (lr_max - lr_min) × (1 + cos(π×T_cur/T_max)) / 2

       像余弦曲线一样平滑下降

    4. Warm-up（预热）
       开始时学习率从很小逐渐增大

       为什么？
       - 开始时参数是随机的
       - 大学习率可能导致梯度爆炸
       - 先用小学习率"热身"

    ====================================================================
    """

    @staticmethod
    def step_decay(initial_lr, epoch, step_size=10, gamma=0.5):
        """阶梯衰减"""
        return initial_lr * (gamma ** (epoch // step_size))

    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """指数衰减"""
        return initial_lr * (decay_rate ** epoch)

    @staticmethod
    def cosine_annealing(initial_lr, epoch, T_max, eta_min=0):
        """余弦退火"""
        return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2

    @staticmethod
    def linear_warmup(initial_lr, epoch, warmup_epochs):
        """线性预热"""
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        return initial_lr


def visualize_lr_schedules():
    """可视化不同的学习率调度策略"""
    print("\n" + "=" * 70)
    print("Visualization: Learning Rate Schedules")
    print("=" * 70)

    initial_lr = 0.1
    n_epochs = 100
    epochs = np.arange(n_epochs)

    # 计算不同策略的学习率
    schedules = {
        'Constant': [initial_lr] * n_epochs,
        'Step Decay': [LearningRateScheduler.step_decay(initial_lr, e, step_size=20)
                      for e in epochs],
        'Exponential': [LearningRateScheduler.exponential_decay(initial_lr, e, decay_rate=0.95)
                       for e in epochs],
        'Cosine': [LearningRateScheduler.cosine_annealing(initial_lr, e, T_max=n_epochs)
                  for e in epochs],
        'Warmup+Decay': [LearningRateScheduler.linear_warmup(initial_lr, e, warmup_epochs=10)
                        if e < 10 else
                        LearningRateScheduler.exponential_decay(initial_lr, e-10, decay_rate=0.96)
                        for e in epochs],
    }

    # 可视化
    plt.figure(figsize=(12, 7))

    colors = ['gray', 'blue', 'green', 'red', 'purple']
    for (name, lr_values), color in zip(schedules.items(), colors):
        plt.plot(epochs, lr_values, label=name, linewidth=2.5, alpha=0.8, color=color)

    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Learning Rate', fontsize=11)
    plt.title('Learning Rate Scheduling Strategies', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_schedules.png', dpi=100, bbox_inches='tight')
    print("\n📊 LR schedules saved to: lr_schedules.png")
    plt.close()

    print("\n💡 When to Use:")
    print("  - Constant: Baseline, simple")
    print("  - Step Decay: Good for stable training")
    print("  - Exponential: Smooth decay")
    print("  - Cosine: Popular in modern training (smooth)")
    print("  - Warmup: Essential for large batch size or unstable start")


# ==================== 3. Regularization Techniques ====================
def demo_dropout():
    """
    Dropout 演示

    ====================================================================
    🔑 What is Dropout?
    ====================================================================

    Dropout = 训练时随机"关闭"一些神经元

    工作原理：
        训练时：
            - 每次前向传播，随机将一部分神经元输出设为0
            - dropout_rate = 0.5 → 50%的神经元被关闭

        测试时：
            - 所有神经元都工作
            - 输出乘以 (1 - dropout_rate) 来缩放

    ====================================================================
    🔑 为什么 Dropout 有效？
    ====================================================================

    1. 防止神经元共适应（co-adaptation）
       - 没有dropout：某些神经元总是一起工作
       - 有dropout：神经元不能依赖特定的其他神经元
       - 每个神经元必须学到更鲁棒的特征

    2. 类似集成学习（ensemble）
       - 每次dropout产生一个不同的子网络
       - 训练了很多个子网络的集合
       - 测试时相当于平均所有子网络

    3. 类比：
       - 像一个团队，不能总是依赖某几个人
       - 每个人都要学会独立工作
       - 团队才更robust

    ====================================================================
    🔑 Dropout Rate 选择
    ====================================================================

    dropout_rate = 0.0:  没有dropout
    dropout_rate = 0.2:  轻度正则化
    dropout_rate = 0.5:  常用值（丢弃一半）
    dropout_rate = 0.8:  重度正则化（可能欠拟合）

    经验：
    - 全连接层：0.5
    - 卷积层：0.2-0.3（卷积本身有正则化效果）
    - RNN：更小（0.2），否则影响记忆

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("Demo: Dropout Regularization")
    print("=" * 70)

    # 生成过拟合场景的数据
    np.random.seed(42)
    X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

    print("\nTraining with and without Dropout...")
    print("  Dataset: 100 samples (small, prone to overfitting)")
    print("  Model: 2-layer NN with 20 hidden neurons")

    # 可视化dropout效果
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制数据
    for ax in axes:
        ax.scatter(X[y==0, 0], X[y==0, 1], c='red', s=50, alpha=0.6, label='Class 0')
        ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=50, alpha=0.6, label='Class 1')
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_title('Without Dropout\n(May overfit)', fontsize=12, fontweight='bold')
    axes[1].set_title('With Dropout (rate=0.5)\n(Better generalization)',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('dropout_demo.png', dpi=100, bbox_inches='tight')
    print("\n📊 Dropout demo saved to: dropout_demo.png")
    plt.close()

    print("\n💡 Key Points:")
    print("  - Dropout randomly drops neurons during training")
    print("  - Forces network to learn redundant representations")
    print("  - Acts like training multiple networks (ensemble)")
    print("  - Must disable dropout during testing!")


def demo_batch_normalization():
    """
    Batch Normalization 演示

    ====================================================================
    🔑 What is Batch Normalization?
    ====================================================================

    BatchNorm = 对每一层的输入进行归一化

    工作原理：
        对于一个mini-batch的数据：
        1. 计算均值和方差
        2. 归一化：(x - mean) / std
        3. 缩放和平移：γ × x_norm + β

    公式：
        μ = (1/m) Σ xᵢ            # 批次均值
        σ² = (1/m) Σ (xᵢ - μ)²   # 批次方差
        x̂ = (x - μ) / √(σ² + ε)  # 归一化
        y = γ × x̂ + β             # 缩放平移

    ====================================================================
    🔑 为什么 BatchNorm 有效？
    ====================================================================

    1. 解决内部协变量偏移（Internal Covariate Shift）
       - 每一层的输入分布不断变化
       - BatchNorm让每一层的输入保持稳定

    2. 允许更大的学习率
       - 归一化后梯度更稳定
       - 不容易梯度爆炸

    3. 起到轻微的正则化作用
       - 每个batch的统计量有随机性
       - 类似加了噪声

    4. 减少对初始化的依赖
       - 即使初始化不好，BatchNorm也能拉回来

    ====================================================================
    🔑 使用注意事项
    ====================================================================

    训练时：
        - 使用当前batch的均值和方差
        - 更新运行时的移动平均（用于测试）

    测试时：
        - 使用训练时的移动平均统计量
        - 保证测试时的确定性

    放置位置：
        - 通常放在激活函数之前
        - Conv → BatchNorm → ReLU
        - Linear → BatchNorm → ReLU

    ====================================================================
    """
    print("\n" + "=" * 70)
    print("Demo: Batch Normalization")
    print("=" * 70)

    print("\nBatch Normalization normalizes layer inputs")
    print("  Benefits:")
    print("    1. Faster convergence")
    print("    2. Higher learning rates possible")
    print("    3. Less sensitive to initialization")
    print("    4. Slight regularization effect")

    # 可视化BatchNorm的效果
    np.random.seed(42)

    # 模拟一个batch的数据（未归一化）
    batch_size = 32
    features = 100
    x = np.random.randn(batch_size, features) * 5 + 10  # 均值10，标准差5

    # BatchNorm
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x_norm = (x - mean) / (std + 1e-8)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原始分布
    axes[0].hist(x.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0].axvline(x.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean={x.mean():.2f}')
    axes[0].set_xlabel('Value', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Before Batch Normalization\n(Mean≠0, Std≠1)',
                     fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # BatchNorm后的分布
    axes[1].hist(x_norm.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x_norm.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean={x_norm.mean():.2f}')
    axes[1].set_xlabel('Value', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('After Batch Normalization\n(Mean≈0, Std≈1)',
                     fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('batch_norm_demo.png', dpi=100, bbox_inches='tight')
    print("\n📊 BatchNorm demo saved to: batch_norm_demo.png")
    plt.close()

    print("\n💡 Key Points:")
    print("  - Normalizes each layer's inputs to mean=0, std=1")
    print("  - Stabilizes training, enables higher learning rates")
    print("  - Almost always used in modern deep networks")


# ==================== 4. Main Program ====================
def main():
    print("=" * 70)
    print("Optimization and Regularization Techniques")
    print("=" * 70)

    # 1. Compare optimizers
    compare_optimizers()

    # 2. Learning rate schedules
    visualize_lr_schedules()

    # 3. Dropout
    demo_dropout()

    # 4. Batch Normalization
    demo_batch_normalization()

    # 5. Summary
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Optimizers (优化器)

   SGD (Stochastic Gradient Descent):
   - θ = θ - lr × ∇loss
   - 简单但慢，容易卡住
   - 学习率固定，所有参数用同一个

   Momentum (动量):
   - velocity = β × velocity - lr × ∇loss
   - θ = θ + velocity
   - 像滚球，有惯性，可以冲过小坑
   - β=0.9常用

   RMSprop (均方根传播):
   - cache = β × cache + (1-β) × (∇loss)²
   - θ = θ - lr × ∇loss / √cache
   - 自适应学习率，对不同参数用不同步长
   - β=0.9常用

   Adam (自适应矩估计):
   - 结合 Momentum + RMSprop
   - m = β₁ × m + (1-β₁) × ∇loss      (一阶矩)
   - v = β₂ × v + (1-β₂) × (∇loss)²   (二阶矩)
   - θ = θ - lr × m / √v
   - 🌟最流行！默认选择
   - β₁=0.9, β₂=0.999, lr=0.001常用

2. Learning Rate Scheduling (学习率调度)

   为什么需要？
   - 固定学习率：开始太慢或太震荡
   - 动态调整：开始大步走，后期小步调

   常见策略：

   Step Decay (阶梯衰减):
   - 每N个epoch减半
   - 例：lr = 0.1 → 0.05 → 0.025

   Exponential Decay (指数衰减):
   - lr = lr₀ × decay_rate^epoch
   - 平滑下降

   Cosine Annealing (余弦退火):
   - 像余弦曲线平滑下降
   - 现代流行

   Warm-up (预热):
   - 开始时从很小的学习率逐渐增大
   - 防止开始时梯度爆炸
   - 大batch训练必备

3. Regularization (正则化) - 防止过拟合

   L2 Regularization (权重衰减):
   - loss = loss + λ × Σ(weights²)
   - 惩罚大权重，让模型更简单
   - λ=0.01或0.001常用

   Dropout:
   - 训练时随机关闭一些神经元
   - 防止神经元共适应
   - rate=0.5常用（FC层），0.2（Conv层）
   - 🚫测试时必须关闭！

   Batch Normalization:
   - 归一化每层的输入：mean=0, std=1
   - 加速训练，提高稳定性
   - 轻微正则化效果
   - 🌟几乎总是使用

   Early Stopping:
   - 监控验证集loss
   - 不再下降就停止
   - 简单有效

4. Gradient Clipping (梯度裁剪)

   为什么？
   - 防止梯度爆炸（尤其RNN）

   方法：
   - if ||gradient|| > threshold:
       gradient = threshold × gradient / ||gradient||

   使用：
   - threshold=1.0或5.0常用
   - RNN必备

5. Weight Initialization (权重初始化)

   为什么重要？
   - 初始化不好 → 梯度消失/爆炸

   常见方法：

   Xavier/Glorot:
   - 适合 Sigmoid/Tanh
   - W ~ U(-√(6/(n_in+n_out)), √(6/(n_in+n_out)))

   He Initialization:
   - 适合 ReLU
   - W ~ N(0, √(2/n_in))
   - 🌟ReLU网络默认选择

6. 实战技巧

   训练深度网络的标准配方：

   1. 优化器：Adam (lr=0.001)
   2. BatchNorm：每层Conv/FC后加
   3. Dropout：FC层加0.5，Conv层加0.2-0.3
   4. 初始化：ReLU用He，Sigmoid用Xavier
   5. 学习率：Cosine或Step Decay
   6. Warm-up：大batch时使用
   7. 梯度裁剪：RNN必须，其他可选
   8. Early Stopping：监控验证集

   调参顺序：
   1. 先用Adam默认参数
   2. 调学习率（0.0001~0.01）
   3. 加BatchNorm
   4. 如果过拟合，加Dropout
   5. 如果还不行，加L2正则化
   6. 最后考虑学习率调度

7. 常见问题诊断

   Loss不下降：
   - 学习率太小 → 增大
   - 学习率太大 → 减小
   - 梯度消失 → 加BatchNorm，换ReLU
   - 初始化不好 → 用He/Xavier

   Loss震荡：
   - 学习率太大 → 减小
   - Batch size太小 → 增大

   过拟合（训练好，测试差）：
   - 加Dropout
   - 加L2正则化
   - 早停
   - 增加数据

   欠拟合（训练也不好）：
   - 模型太简单 → 加层/加神经元
   - 正则化太强 → 减小Dropout/L2
   - 学习率太小 → 增大

   梯度爆炸：
   - 学习率太大 → 减小
   - 加梯度裁剪
   - 加BatchNorm
   - 检查初始化

8. 优化器选择指南

   默认选择：
   - 🌟Adam：99%情况下都好用

   特殊情况：
   - 计算资源有限：SGD with Momentum
   - 需要最好泛化：SGD with Momentum + 学习率调度
   - RNN/LSTM：Adam或RMSprop
   - GAN：RMSprop或Adam (β₁=0.5)
   - Transformer：Adam + Warmup + Cosine Decay

9. 超参数范围参考

   Learning Rate:
   - Adam: 0.0001 ~ 0.01 (default 0.001)
   - SGD: 0.01 ~ 0.1 (default 0.01)

   Batch Size:
   - 小数据集: 16 ~ 64
   - 大数据集: 128 ~ 512
   - 越大越稳定，但需要Warmup

   Dropout Rate:
   - FC层: 0.5
   - Conv层: 0.2 ~ 0.3
   - RNN: 0.2

   L2 Regularization:
   - λ = 0.0001 ~ 0.01 (default 0.001)

   Gradient Clipping:
   - threshold = 1.0 ~ 5.0

10. 记住
    - 没有万能的超参数组合
    - 多实验，多观察曲线
    - 从简单开始（SGD → Adam → +技巧）
    - 优先解决过拟合/欠拟合
    - 优化器和正则化是两回事（目的不同）
    """)


if __name__ == "__main__":
    main()

    print("\n💡 Practice Suggestions:")
    print("  1. Implement gradient descent with different optimizers")
    print("  2. Compare training curves with/without BatchNorm")
    print("  3. Tune learning rate to see its impact")
    print("  4. Experiment with different Dropout rates")
    print("  5. Train a network on MNIST with these techniques")

