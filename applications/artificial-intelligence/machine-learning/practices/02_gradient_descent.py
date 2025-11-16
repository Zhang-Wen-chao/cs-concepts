"""
梯度下降算法对比 (Gradient Descent Variants)

问题：对比三种梯度下降方法的优缺点
目标：理解 BGD、SGD、Mini-batch GD 的差异和适用场景

核心概念：
1. BGD (Batch GD)：每次用全部数据计算梯度 → 稳定但慢
2. SGD (Stochastic GD)：每次用1个样本计算梯度 → 快但波动大
3. Mini-batch GD：每次用一小批样本 → 折中方案（最常用）
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================== 1. 准备数据 ====================
def generate_data(n_samples=100):
    """
    生成模拟数据：房子面积 -> 房价
    真实关系：房价 = 3 * 面积 + 2 + 噪声
    """
    np.random.seed(42)
    X = np.random.uniform(1, 5, n_samples)
    y = 3 * X + 2 + np.random.normal(0, 0.5, n_samples)
    return X, y


# ==================== 2. 三种梯度下降实现 ====================
class GradientDescentComparison:
    """对比三种梯度下降方法"""

    def __init__(self, learning_rate=0.01, n_epochs=100):
        """
        参数：
            learning_rate: 学习率
            n_epochs: 训练轮数（epoch = 遍历完整个数据集一次）
        """
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None

    def _compute_gradient(self, X, y, w, b):
        """
        计算梯度（通用函数）

        参数：
            X, y: 数据（可以是全部、一个样本、或一批样本）
            w, b: 当前参数
        返回：
            dw, db: 梯度
        """
        n = len(X)
        y_pred = w * X + b
        dw = (2 / n) * np.sum((y_pred - y) * X)
        db = (2 / n) * np.sum(y_pred - y)
        return dw, db

    def _compute_loss(self, X, y, w, b):
        """计算 MSE 损失"""
        y_pred = w * X + b
        return np.mean((y_pred - y) ** 2)

    # ========== 方法1：批量梯度下降 (BGD) ==========
    def batch_gd(self, X, y):
        """
        批量梯度下降：每次用全部数据计算梯度

        优点：
        - 梯度准确，下降稳定
        - 理论收敛保证强

        缺点：
        - 数据量大时很慢（每次迭代要计算所有样本）
        - 内存占用大
        """
        n_samples = len(X)
        self.w, self.b = 0.0, 0.0
        loss_history = []
        w_history, b_history = [self.w], [self.b]

        for epoch in range(self.n_epochs):
            # 用全部数据计算梯度
            dw, db = self._compute_gradient(X, y, self.w, self.b)

            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 记录
            loss = self._compute_loss(X, y, self.w, self.b)
            loss_history.append(loss)
            w_history.append(self.w)
            b_history.append(self.b)

        return loss_history, w_history, b_history

    # ========== 方法2：随机梯度下降 (SGD) ==========
    def stochastic_gd(self, X, y):
        """
        随机梯度下降：每次只用1个样本计算梯度

        优点：
        - 更新频繁，收敛快
        - 内存占用小
        - 可能跳出局部最优

        缺点：
        - 梯度噪声大，波动剧烈
        - 不保证每次迭代都减小损失
        - 需要调整学习率（通常要更小）
        """
        n_samples = len(X)
        self.w, self.b = 0.0, 0.0
        loss_history = []
        w_history, b_history = [self.w], [self.b]

        for epoch in range(self.n_epochs):
            # 打乱数据顺序（重要！避免顺序偏差）
            indices = np.random.permutation(n_samples)

            # 遍历每个样本
            for i in indices:
                # 用单个样本计算梯度
                X_i = np.array([X[i]])
                y_i = np.array([y[i]])
                dw, db = self._compute_gradient(X_i, y_i, self.w, self.b)

                # 更新参数
                self.w -= self.lr * dw
                self.b -= self.lr * db

            # 记录（每个 epoch 结束后）
            loss = self._compute_loss(X, y, self.w, self.b)
            loss_history.append(loss)
            w_history.append(self.w)
            b_history.append(self.b)

        return loss_history, w_history, b_history

    # ========== 方法3：小批量梯度下降 (Mini-batch GD) ==========
    def minibatch_gd(self, X, y, batch_size=10):
        """
        小批量梯度下降：每次用一小批样本计算梯度

        优点：
        - 平衡了 BGD 和 SGD 的优缺点
        - 可以利用向量化加速
        - 梯度估计相对准确且更新频繁
        - 工业界最常用（深度学习默认选择）

        缺点：
        - 需要调整 batch_size 这个超参数

        batch_size 选择建议：
        - 小数据集（<1000）：32-64
        - 中数据集（1000-10万）：128-256
        - 大数据集（>10万）：256-512
        """
        n_samples = len(X)
        self.w, self.b = 0.0, 0.0
        loss_history = []
        w_history, b_history = [self.w], [self.b]

        for epoch in range(self.n_epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)

            # 分批处理
            for start_idx in range(0, n_samples, batch_size):
                # 取一个 batch
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # 用 batch 计算梯度
                dw, db = self._compute_gradient(X_batch, y_batch, self.w, self.b)

                # 更新参数
                self.w -= self.lr * dw
                self.b -= self.lr * db

            # 记录
            loss = self._compute_loss(X, y, self.w, self.b)
            loss_history.append(loss)
            w_history.append(self.w)
            b_history.append(self.b)

        return loss_history, w_history, b_history

    def predict(self, X):
        """预测"""
        return self.w * X + self.b


# ==================== 3. 可视化对比 ====================
def visualize_comparison(X, y, results):
    """
    可视化三种方法的对比

    results: {
        'BGD': (loss_history, w_history, b_history),
        'SGD': (...),
        'Mini-batch': (...)
    }
    """
    fig = plt.figure(figsize=(18, 5))

    # 子图1：损失函数下降曲线
    plt.subplot(1, 3, 1)
    for name, (loss_history, _, _) in results.items():
        plt.plot(loss_history, label=name, linewidth=2, alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Loss vs Epoch (Convergence Speed)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 对数坐标，方便观察

    # 子图2：参数收敛轨迹（在参数空间中的路径）
    plt.subplot(1, 3, 2)
    true_w, true_b = 3, 2  # 真实值
    plt.scatter(true_w, true_b, s=300, c='red', marker='*',
                label='True (w=3, b=2)', zorder=5, edgecolors='black', linewidths=2)

    for name, (_, w_history, b_history) in results.items():
        plt.plot(w_history, b_history, 'o-', label=name, alpha=0.7, markersize=4)

    plt.xlabel('Weight (w)', fontsize=12)
    plt.ylabel('Bias (b)', fontsize=12)
    plt.title('Parameter Trajectory in (w, b) Space', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 子图3：最终拟合效果
    plt.subplot(1, 3, 3)
    plt.scatter(X, y, alpha=0.5, label='Data', s=20)

    colors = ['red', 'green', 'blue']
    for (name, (_, w_history, b_history)), color in zip(results.items(), colors):
        final_w, final_b = w_history[-1], b_history[-1]
        X_line = np.linspace(X.min(), X.max(), 100)
        y_line = final_w * X_line + final_b
        plt.plot(X_line, y_line, color=color, linewidth=2,
                label=f'{name}: y={final_w:.2f}x+{final_b:.2f}', alpha=0.8)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Final Fitted Lines', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_descent_comparison.png', dpi=100)
    print("\n📊 对比图已保存到: gradient_descent_comparison.png")
    plt.show()


# ==================== 4. 主程序 ====================
def main():
    print("=" * 70)
    print("梯度下降算法对比：BGD vs SGD vs Mini-batch GD")
    print("=" * 70)

    # 生成数据
    print("\n📊 生成数据（100个样本，真实关系：y = 3x + 2）")
    X, y = generate_data(100)

    # 训练参数
    learning_rate = 0.01
    n_epochs = 50

    print(f"\n⚙️  训练参数：")
    print(f"   学习率: {learning_rate}")
    print(f"   训练轮数: {n_epochs}")
    print(f"   Mini-batch 大小: 10")

    # 训练三种方法
    results = {}

    print("\n" + "-" * 70)
    print("🔵 方法1：批量梯度下降 (BGD)")
    print("   策略：每次用全部100个样本计算梯度")
    model_bgd = GradientDescentComparison(learning_rate, n_epochs)
    loss_bgd, w_bgd, b_bgd = model_bgd.batch_gd(X, y)
    results['BGD'] = (loss_bgd, w_bgd, b_bgd)
    print(f"   最终结果：w={model_bgd.w:.4f}, b={model_bgd.b:.4f}, loss={loss_bgd[-1]:.4f}")

    print("\n" + "-" * 70)
    print("🟢 方法2：随机梯度下降 (SGD)")
    print("   策略：每次只用1个样本计算梯度")
    model_sgd = GradientDescentComparison(learning_rate, n_epochs)
    loss_sgd, w_sgd, b_sgd = model_sgd.stochastic_gd(X, y)
    results['SGD'] = (loss_sgd, w_sgd, b_sgd)
    print(f"   最终结果：w={model_sgd.w:.4f}, b={model_sgd.b:.4f}, loss={loss_sgd[-1]:.4f}")

    print("\n" + "-" * 70)
    print("🟡 方法3：小批量梯度下降 (Mini-batch GD)")
    print("   策略：每次用10个样本计算梯度")
    model_minibatch = GradientDescentComparison(learning_rate, n_epochs)
    loss_minibatch, w_minibatch, b_minibatch = model_minibatch.minibatch_gd(X, y, batch_size=10)
    results['Mini-batch'] = (loss_minibatch, w_minibatch, b_minibatch)
    print(f"   最终结果：w={model_minibatch.w:.4f}, b={model_minibatch.b:.4f}, loss={loss_minibatch[-1]:.4f}")

    # 对比分析
    print("\n" + "=" * 70)
    print("📊 对比分析")
    print("=" * 70)
    print(f"\n收敛速度（最终损失）：")
    print(f"  BGD:        {loss_bgd[-1]:.6f}")
    print(f"  SGD:        {loss_sgd[-1]:.6f}")
    print(f"  Mini-batch: {loss_minibatch[-1]:.6f}")

    print(f"\n参数准确度（真实值 w=3, b=2）：")
    print(f"  BGD:        w={model_bgd.w:.4f}, b={model_bgd.b:.4f}")
    print(f"  SGD:        w={model_sgd.w:.4f}, b={model_sgd.b:.4f}")
    print(f"  Mini-batch: w={model_minibatch.w:.4f}, b={model_minibatch.b:.4f}")

    # 可视化
    print("\n🎨 生成可视化图表...")
    visualize_comparison(X, y, results)

    # 总结
    print("\n" + "=" * 70)
    print("✅ 核心结论")
    print("=" * 70)
    print("""
1. BGD (批量梯度下降)
   ✓ 最稳定，梯度最准确
   ✗ 大数据集时很慢
   💡 适用：小数据集、需要精确收敛

2. SGD (随机梯度下降)
   ✓ 更新最快，内存占用小
   ✗ 波动大，不稳定
   💡 适用：超大数据集、在线学习

3. Mini-batch GD (小批量梯度下降) ⭐ 推荐
   ✓ 平衡速度和稳定性
   ✓ 可以利用GPU并行加速
   ✓ 深度学习默认选择
   💡 适用：几乎所有场景（工业界标准）

关键建议：
- 数据集 <1000：用 BGD 或 Mini-batch(32-64)
- 数据集 >10万：必须用 Mini-batch(128-512) 或 SGD
- 深度学习：始终用 Mini-batch GD
    """)

    print("=" * 70)


# ==================== 5. 实验区 ====================
def experiment_batch_size():
    """
    实验：不同 batch size 的影响

    观察从 SGD (batch=1) 到 BGD (batch=100) 的过渡
    """
    print("\n" + "=" * 70)
    print("🧪 实验：不同 Batch Size 的影响")
    print("=" * 70)

    X, y = generate_data(100)
    learning_rate = 0.01
    n_epochs = 50

    batch_sizes = [1, 5, 10, 20, 50, 100]
    results = {}

    for bs in batch_sizes:
        model = GradientDescentComparison(learning_rate, n_epochs)
        if bs == 100:
            # batch_size = 100 = 数据总数，等同于 BGD
            loss_history, w_history, b_history = model.batch_gd(X, y)
            name = f'Batch={bs} (BGD)'
        else:
            loss_history, w_history, b_history = model.minibatch_gd(X, y, batch_size=bs)
            name = f'Batch={bs}'

        results[name] = (loss_history, w_history, b_history)
        print(f"  {name:15s}: 最终损失={loss_history[-1]:.6f}, w={model.w:.4f}, b={model.b:.4f}")

    # 可视化
    plt.figure(figsize=(14, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    for name, (loss_history, _, _) in results.items():
        plt.plot(loss_history, label=name, linewidth=2, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss vs Epoch for Different Batch Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 参数轨迹
    plt.subplot(1, 2, 2)
    plt.scatter(3, 2, s=300, c='red', marker='*', label='True (3, 2)', zorder=5)
    for name, (_, w_history, b_history) in results.items():
        plt.plot(w_history, b_history, 'o-', label=name, alpha=0.7, markersize=3)
    plt.xlabel('Weight (w)')
    plt.ylabel('Bias (b)')
    plt.title('Parameter Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('batch_size_comparison.png', dpi=100)
    print("\n📊 Batch size 对比图已保存到: batch_size_comparison.png")
    plt.show()

    print("\n💡 观察：")
    print("  - Batch size 越小 → 更新越频繁，但波动越大")
    print("  - Batch size 越大 → 更稳定，但更新越慢")
    print("  - Mini-batch (10-50) → 最佳平衡点")


if __name__ == "__main__":
    # 主实验
    main()

    # 可选：Batch size 实验（取消注释运行）
    experiment_batch_size()

    print("\n💡 提示：")
    print("  - 取消 experiment_batch_size() 的注释，探索 batch size 的影响")
    print("  - 尝试修改学习率，观察三种方法的表现差异")
    print("  - 思考：为什么深度学习几乎总是用 Mini-batch GD？")
