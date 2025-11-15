"""
线性回归 (Linear Regression) - 从零实现

问题：根据房子面积预测房价
目标：理解监督学习的基本流程和梯度下降算法

核心概念：
1. 模型：y = w * x + b  (w=权重, b=偏置)
2. 损失函数：MSE (Mean Squared Error, 均方误差) = 1/n * Σ(预测值 - 真实值)²
3. 优化：通过梯度下降最小化损失函数
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
    # 面积：1-5（归一化到较小范围，避免梯度爆炸）
    X = np.random.uniform(1, 5, n_samples)
    # 房价 = 3 * 面积 + 2 + 噪声
    y = 3 * X + 2 + np.random.normal(0, 0.5, n_samples)
    return X, y


# ==================== 2. 线性回归模型 ====================
class LinearRegression:
    """从零实现的线性回归"""

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        """
        参数：
            learning_rate: 学习率（步长）
            n_iterations: 训练迭代次数
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = None  # 权重
        self.b = None  # 偏置
        self.loss_history = []  # 记录损失变化

    def fit(self, X, y):
        """
        训练模型：通过梯度下降找到最优的 w 和 b

        数学原理：
        - 预测：y_pred = w * x + b
        - 损失：loss = 1/n * Σ(y_pred - y)²
        - 梯度：dw = 2/n * Σ(y_pred - y) * x
        -       db = 2/n * Σ(y_pred - y)
        - 更新：w = w - lr * dw
        -       b = b - lr * db
        """
        n_samples = len(X)

        # 初始化参数（随机初始化）
        self.w = 0.0
        self.b = 0.0

        # 梯度下降训练
        for i in range(self.n_iterations):
            # 前向传播：计算预测值
            y_pred = self.w * X + self.b

            # 计算损失（MSE）
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # 计算梯度
            dw = (2 / n_samples) * np.sum((y_pred - y) * X)
            db = (2 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 每100次打印一次
            if (i + 1) % 100 == 0:
                print(f"第 {i+1} 次迭代 - 损失: {loss:.2f}, w={self.w:.4f}, b={self.b:.4f}")

    def predict(self, X):
        """预测"""
        return self.w * X + self.b


# ==================== 3. 可视化 ====================
def plot_results(X, y, model):
    """可视化训练结果"""
    plt.figure(figsize=(15, 5))

    # 子图1：数据和拟合线
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, alpha=0.5, label='Real Data')
    plt.plot(X, model.predict(X), 'r-', linewidth=2, label='Fitted Line')
    plt.xlabel('Area (X)')
    plt.ylabel('Price (Y)')
    plt.title('Linear Regression Result')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：损失函数变化
    plt.subplot(1, 3, 2)
    plt.plot(model.loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (MSE)')
    plt.title('Training: Loss Decreasing')
    plt.grid(True, alpha=0.3)

    # 子图3：前100次迭代的损失（放大看）
    plt.subplot(1, 3, 3)
    plt.plot(model.loss_history[:100])
    plt.xlabel('Iterations')
    plt.ylabel('Loss (MSE)')
    plt.title('First 100 Iterations (Zoomed)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('linear_regression_result.png', dpi=100)
    print("\n图表已保存到: linear_regression_result.png")
    plt.show()


# ==================== 4. 主程序 ====================
def main():
    print("=" * 60)
    print("线性回归实践：从零实现")
    print("=" * 60)

    # 生成数据
    print("\n1. 生成模拟数据（100个样本）")
    print("   真实关系：房价 = 3 * 面积 + 2 + 噪声")
    X, y = generate_data(100)
    print(f"   数据范围 - 面积: {X.min():.2f}-{X.max():.2f}")
    print(f"            房价: {y.min():.2f}-{y.max():.2f}")

    # 训练模型
    print("\n2. 训练线性回归模型")
    print("   学习率: 0.001, 迭代次数: 1000")
    print("-" * 60)
    model = LinearRegression(learning_rate=0.001, n_iterations=1000)
    model.fit(X, y)

    # 查看学到的参数
    print("\n3. 模型学到的参数")
    print(f"   权重 w = {model.w:.4f}  (真实值约为 3)")
    print(f"   偏置 b = {model.b:.4f}  (真实值约为 2)")
    print(f"   最终损失 = {model.loss_history[-1]:.2f}")

    # 预测新样本
    print("\n4. 用训练好的模型预测")
    test_areas = [2.0, 3.0, 4.0]
    for area in test_areas:
        price = model.predict(np.array([area]))[0]
        expected = 3 * area + 2
        print(f"   面积 {area:.1f} → 预测: {price:.2f}, 期望: {expected:.2f}")

    # 可视化
    print("\n5. 生成可视化图表...")
    plot_results(X, y, model)

    print("\n" + "=" * 60)
    print("✅ 完成！你已经理解了：")
    print("   1. 监督学习的基本流程（数据 → 训练 → 预测）")
    print("   2. 损失函数的概念（衡量预测误差）")
    print("   3. 梯度下降的优化过程（逐步减小损失）")
    print("=" * 60)


# ==================== 5. 实验区 ====================
def experiment():
    """
    实验：观察不同学习率的影响

    第一轮实验结果（500次迭代，真实值 w≈3, b≈2）：
    - lr=0.0001: 损失15.55, w=2.15, b=0.66 → 太慢，未收敛 ✗
    - lr=0.001:  损失0.35,  w=3.25, b=1.14 → 收敛良好 ✓
    - lr=0.01:   损失0.22,  w=3.04, b=1.85 → 收敛最好 ✓✓

    第二轮实验结果（测试更大学习率）：
    - lr=0.01:   损失0.22,  w=3.04, b=1.85 → 稳定收敛 ✓
    - lr=0.05:   损失0.20,  w=2.94, b=2.16 → 最优！快速且准确 ✓✓✓
    - lr=0.1:    损失爆炸,  w=-7e23, b=-2e23 → 梯度爆炸，完全发散 ✗✗

    结论：lr=0.05 是最佳学习率（快速收敛+最接近真实值），lr=0.1 太大导致发散

    尝试修改这些参数，看看会发生什么：
    - learning_rate: 0.0001, 0.001, 0.01, 0.1
    - n_iterations: 100, 1000, 5000
    """
    print("\n" + "=" * 60)
    print("🧪 实验：不同学习率的对比")
    print("=" * 60)

    X, y = generate_data(100)
    # learning_rates = [0.0001, 0.001, 0.01]  # 第一轮
    learning_rates = [0.01, 0.05, 0.1]  # 第二轮：测试更大学习率

    plt.figure(figsize=(12, 4))

    for idx, lr in enumerate(learning_rates, 1):
        model = LinearRegression(learning_rate=lr, n_iterations=500)
        model.fit(X, y)

        plt.subplot(1, 3, idx)
        plt.plot(model.loss_history)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(f'Learning Rate = {lr}')
        plt.grid(True, alpha=0.3)

        print(f"\n学习率 {lr}:")
        print(f"  最终损失: {model.loss_history[-1]:.2f}")
        print(f"  学到的参数: w={model.w:.4f}, b={model.b:.4f}")

    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=100)
    print("\n对比图已保存到: learning_rate_comparison.png")
    plt.show()


def experiment_lr_sweep():
    """
    实验：学习率扫描 - 找到最优学习率

    在一定范围内测试多个学习率，绘制：
    1. 学习率 vs 最终损失曲线（找到最优点）
    2. 不同学习率的训练过程对比
    """
    print("\n" + "=" * 60)
    print("🔍 实验：学习率扫描（Learning Rate Sweep）")
    print("=" * 60)

    X, y = generate_data(100)

    # 测试更密集的学习率范围
    learning_rates = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    n_iterations = 500

    # 存储结果
    results = []

    # 训练每个学习率
    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr, n_iterations=n_iterations)
        model.fit(X, y)

        final_loss = model.loss_history[-1]
        results.append({
            'lr': lr,
            'loss': final_loss,
            'w': model.w,
            'b': model.b,
            'history': model.loss_history,
            'converged': final_loss < 100  # 判断是否收敛
        })

        status = "✓ 收敛" if final_loss < 100 else "✗ 发散"
        print(f"lr={lr:6.3f}: 损失={final_loss:12.2f}, w={model.w:8.4f}, b={model.b:8.4f} {status}")

    # 找到最优学习率（在收敛的结果中）
    converged_results = [r for r in results if r['converged']]
    if converged_results:
        best = min(converged_results, key=lambda x: x['loss'])
        print(f"\n🎯 最优学习率: {best['lr']}, 损失={best['loss']:.4f}, w={best['w']:.4f}, b={best['b']:.4f}")

    # ==================== 可视化 ====================
    fig = plt.figure(figsize=(16, 5))

    # 子图1：学习率 vs 最终损失（关键图）
    plt.subplot(1, 3, 1)
    lrs = [r['lr'] for r in results]
    losses = [r['loss'] for r in results]

    # 分离收敛和发散的点
    converged_lrs = [r['lr'] for r in results if r['converged']]
    converged_losses = [r['loss'] for r in results if r['converged']]
    diverged_lrs = [r['lr'] for r in results if not r['converged']]
    diverged_losses = [r['loss'] for r in results if not r['converged']]

    plt.plot(converged_lrs, converged_losses, 'o-', linewidth=2, markersize=8, label='Converged')
    if diverged_lrs:
        plt.plot(diverged_lrs, diverged_losses, 'rx', markersize=10, label='Diverged')

    # 标注最优点
    if converged_results:
        plt.plot(best['lr'], best['loss'], 'g*', markersize=20, label=f"Best: lr={best['lr']}")

    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Final Loss', fontsize=12)
    plt.title('Learning Rate vs Final Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 对数坐标，方便看发散的情况

    # 子图2：训练过程对比（只显示收敛的）
    plt.subplot(1, 3, 2)
    for r in converged_results:
        plt.plot(r['history'], label=f"lr={r['lr']}", alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Process (Converged Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3：参数收敛情况（w 和 b）
    plt.subplot(1, 3, 3)
    true_w, true_b = 3, 2  # 真实值
    ws = [r['w'] for r in converged_results]
    bs = [r['b'] for r in converged_results]
    lrs_conv = [r['lr'] for r in converged_results]

    plt.plot(lrs_conv, ws, 'o-', label='Learned w', markersize=8)
    plt.axhline(y=true_w, color='b', linestyle='--', alpha=0.5, label='True w=3')
    plt.plot(lrs_conv, bs, 's-', label='Learned b', markersize=8)
    plt.axhline(y=true_b, color='orange', linestyle='--', alpha=0.5, label='True b=2')
    plt.xlabel('Learning Rate')
    plt.ylabel('Parameter Value')
    plt.title('Parameters vs Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_rate_sweep.png', dpi=100)
    print("\n📊 图表已保存到: learning_rate_sweep.png")
    plt.show()


if __name__ == "__main__":
    # 基础训练
    main()

    # 可选：运行实验（取消注释下面的行）
    # experiment()  # 对比几个学习率的训练过程
    experiment_lr_sweep()  # 学习率扫描实验（推荐！）

    print("\n💡 提示：")
    print("   - 修改 learning_rate 和 n_iterations 观察变化")
    print("   - experiment(): 对比几个学习率的训练过程")
    print("   - experiment_lr_sweep(): 扫描学习率范围，找最优值（推荐）")
    print("   - 尝试自己添加新的实验！")
