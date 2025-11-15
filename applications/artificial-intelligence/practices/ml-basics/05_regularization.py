"""
正则化 (Regularization)

问题：模型在训练集上表现很好，但在测试集上表现差（过拟合）
目标：通过正则化技术防止过拟合，提高模型的泛化能力

核心概念：
1. 过拟合 (Overfitting)：模型过度拟合训练数据，包括噪声
2. 欠拟合 (Underfitting)：模型太简单，无法捕捉数据的规律
3. L1 正则化 (Lasso)：惩罚权重的绝对值，产生稀疏解
4. L2 正则化 (Ridge)：惩罚权重的平方，权重衰减
5. 正则化强度 λ (lambda)：控制正则化的程度

关键思想：在损失函数中添加对权重大小的惩罚
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# ==================== 1. 核心概念 ====================
def add_polynomial_features(X, degree):
    """
    添加多项式特征

    例如：X = [x1, x2]
    degree=1: [x1, x2]
    degree=2: [x1, x2, x1², x1*x2, x2²]
    degree=3: [x1, x2, x1², x1*x2, x2², x1³, x1²*x2, x1*x2², x2³]

    多项式特征可以让线性模型拟合非线性关系
    但也容易导致过拟合（特征太多）
    """
    poly = PolynomialFeatures(degree, include_bias=False)
    return poly.fit_transform(X)


# ==================== 2. 带正则化的线性回归 ====================
class RegularizedLinearRegression:
    """
    带正则化的线性回归

    支持三种模式：
    1. 无正则化（普通线性回归）
    2. L1 正则化（Lasso）
    3. L2 正则化（Ridge）
    """

    def __init__(self, regularization='none', lambda_=0.1, learning_rate=0.01,
                 n_epochs=1000, batch_size=32):
        """
        参数：
            regularization: 'none', 'l1', 'l2'
            lambda_: 正则化强度（λ）
            learning_rate: 学习率
            n_epochs: 训练轮数
            batch_size: 批量大小
        """
        self.regularization = regularization
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        训练模型

        ====================================================================
        🔑 正则化的数学原理
        ====================================================================

        【无正则化】
        Loss = MSE = 1/n * Σ(y - ŷ)²

        【L2 正则化 (Ridge)】
        Loss = MSE + λ * Σw²
             = 1/n * Σ(y - ŷ)² + λ * ||W||²

        惩罚权重的平方和
        - λ = 0：无正则化
        - λ 大：权重被压缩到接近 0
        - 所有权重都变小，但不会变成 0
        - 也叫"权重衰减"（Weight Decay）

        梯度：
        ∂Loss/∂w = ∂MSE/∂w + 2λw
                 = (ŷ - y) * x + 2λw

        【L1 正则化 (Lasso)】
        Loss = MSE + λ * Σ|w|
             = 1/n * Σ(y - ŷ)² + λ * ||W||₁

        惩罚权重的绝对值
        - λ = 0：无正则化
        - λ 大：很多权重被压缩到 0（稀疏解）
        - 可以用于特征选择（权重为 0 = 特征不重要）

        梯度：
        ∂Loss/∂w = ∂MSE/∂w + λ * sign(w)
                 = (ŷ - y) * x + λ * sign(w)

        其中 sign(w) = +1 if w>0, -1 if w<0, 0 if w=0

        ====================================================================
        🔑 L1 vs L2 的直觉理解
        ====================================================================

        想象在二维空间中找最优权重 (w1, w2)

        无正则化：
        - 损失函数是个碗形
        - 最优点在碗底

        L2 正则化：
        - 损失函数 = 原始碗 + 以原点为中心的圆形"山丘"
        - 最优点被"推"向原点
        - 权重变小，但不会变 0
        - 等高线是圆形

        L1 正则化：
        - 损失函数 = 原始碗 + 以原点为中心的菱形"山丘"
        - 菱形的尖角在坐标轴上
        - 最优点容易落在坐标轴上（某个权重 = 0）
        - 等高线是菱形

        ====================================================================
        🔑 为什么正则化可以防止过拟合？
        ====================================================================

        过拟合的特征：
        - 权重很大（对训练数据的小变化非常敏感）
        - 模型过于复杂（高次多项式，特征很多）

        正则化的作用：
        1. 限制权重大小 → 模型更平滑
        2. L1 让一些权重为 0 → 简化模型
        3. 强制模型关注重要特征 → 减少噪声影响

        类比：
        - 无正则化 = 记住所有训练样本的细节（包括噪声）
        - 正则化 = 只记住主要规律，忽略细节

        ====================================================================
        """
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        # Mini-batch 梯度下降
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # 前向传播
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # 计算梯度
                batch_size_actual = len(X_batch)
                error = y_pred - y_batch

                # MSE 的梯度
                dw = (1 / batch_size_actual) * np.dot(X_batch.T, error)
                db = (1 / batch_size_actual) * np.sum(error)

                # 添加正则化项的梯度
                if self.regularization == 'l2':
                    # L2: ∂(λ||W||²)/∂w = 2λw
                    dw += 2 * self.lambda_ * self.weights

                elif self.regularization == 'l1':
                    # L1: ∂(λ||W||₁)/∂w = λ * sign(w)
                    dw += self.lambda_ * np.sign(self.weights)

                # 更新参数
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # 记录损失
            if epoch % 10 == 0:
                y_pred_all = np.dot(X, self.weights) + self.bias
                mse = np.mean((y - y_pred_all) ** 2)

                # 添加正则化项到损失
                if self.regularization == 'l2':
                    reg_term = self.lambda_ * np.sum(self.weights ** 2)
                elif self.regularization == 'l1':
                    reg_term = self.lambda_ * np.sum(np.abs(self.weights))
                else:
                    reg_term = 0

                total_loss = mse + reg_term
                self.loss_history.append(total_loss)

    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """计算 R² 分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# ==================== 3. 数据生成 ====================
def generate_overfitting_data(n_samples=100, noise=10, random_state=42):
    """
    生成容易过拟合的数据

    策略：样本少 + 噪声大 + 后面会加高次多项式特征
    """
    np.random.seed(random_state)

    # 真实关系：y = 2x + 1 + 噪声
    X = np.random.uniform(-3, 3, (n_samples, 1))
    y = 2 * X.ravel() + 1 + np.random.randn(n_samples) * noise

    return X, y


# ==================== 4. 可视化 ====================
def plot_overfitting_demo():
    """
    演示过拟合现象
    """
    print("=" * 70)
    print("📊 演示：什么是过拟合？")
    print("=" * 70)

    # 生成数据
    X_train, y_train = generate_overfitting_data(n_samples=20, noise=2)
    X_test, y_test = generate_overfitting_data(n_samples=100, noise=2, random_state=123)

    # 三种模型：欠拟合、适中、过拟合
    degrees = [1, 3, 15]
    titles = ['Underfitting (度=1)', 'Good Fit (度=3)', 'Overfitting (度=15)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (degree, title) in enumerate(zip(degrees, titles)):
        ax = axes[idx]

        # 添加多项式特征
        X_train_poly = add_polynomial_features(X_train, degree)
        X_test_poly = add_polynomial_features(X_test, degree)

        # 训练模型（无正则化）
        model = RegularizedLinearRegression(regularization='none',
                                            learning_rate=0.01, n_epochs=1000)
        model.fit(X_train_poly, y_train)

        # 评估
        train_score = model.score(X_train_poly, y_train)
        test_score = model.score(X_test_poly, y_test)

        # 绘制
        X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
        X_plot_poly = add_polynomial_features(X_plot, degree)
        y_plot = model.predict(X_plot_poly)

        ax.scatter(X_train, y_train, s=50, alpha=0.7, label='Training data', color='blue')
        ax.plot(X_plot, y_plot, color='red', linewidth=2, label='Model')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'{title}\nTrain R²={train_score:.3f}, Test R²={test_score:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-10, 10)

        # 文字说明
        if idx == 0:
            ax.text(0, -8, '模型太简单\n无法捕捉规律', ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        elif idx == 1:
            ax.text(0, -8, '恰到好处\n泛化能力强', ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            ax.text(0, -8, '过度拟合训练数据\n泛化能力差', ha='center',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.tight_layout()
    plt.savefig('overfitting_demo.png', dpi=100)
    print("\n📊 过拟合演示图已保存到: overfitting_demo.png")
    plt.show()

    print("\n💡 观察：")
    print("  度=1：欠拟合 → 训练和测试 R² 都低（模型太简单）")
    print("  度=3：适中 → 训练和测试 R² 都高（恰到好处）")
    print("  度=15：过拟合 → 训练 R² 高，测试 R² 低（记住了噪声）")


def compare_regularization(X_train, y_train, X_test, y_test, degree=10):
    """
    对比不同正则化方法
    """
    print("\n" + "=" * 70)
    print("🔬 对比：L1 vs L2 vs 无正则化")
    print("=" * 70)

    # 添加多项式特征
    X_train_poly = add_polynomial_features(X_train, degree)
    X_test_poly = add_polynomial_features(X_test, degree)

    print(f"\n多项式度数：{degree}")
    print(f"特征数量：{X_train_poly.shape[1]}")

    # 三种模型
    configs = [
        ('none', 0, 'No Regularization'),
        ('l2', 0.1, 'L2 (Ridge) λ=0.1'),
        ('l1', 0.1, 'L1 (Lasso) λ=0.1')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (reg_type, lambda_, title) in enumerate(configs):
        # 训练模型
        model = RegularizedLinearRegression(
            regularization=reg_type,
            lambda_=lambda_,
            learning_rate=0.01,
            n_epochs=1000
        )
        model.fit(X_train_poly, y_train)

        # 评估
        train_score = model.score(X_train_poly, y_train)
        test_score = model.score(X_test_poly, y_test)

        # 子图1：拟合曲线
        ax1 = axes[0, idx]
        X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
        X_plot_poly = add_polynomial_features(X_plot, degree)
        y_plot = model.predict(X_plot_poly)

        ax1.scatter(X_train, y_train, s=50, alpha=0.7, label='Training', color='blue')
        ax1.scatter(X_test, y_test, s=20, alpha=0.3, label='Test', color='green')
        ax1.plot(X_plot, y_plot, color='red', linewidth=2, label='Model')
        ax1.set_xlabel('X', fontsize=11)
        ax1.set_ylabel('y', fontsize=11)
        ax1.set_title(f'{title}\nTrain R²={train_score:.3f}, Test R²={test_score:.3f}',
                     fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-15, 15)

        # 子图2：权重分布
        ax2 = axes[1, idx]
        weights = model.weights
        ax2.bar(range(len(weights)), weights, alpha=0.7, color='purple')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Feature Index', fontsize=11)
        ax2.set_ylabel('Weight Value', fontsize=11)
        ax2.set_title(f'Weight Distribution\nNon-zero: {np.sum(np.abs(weights) > 0.01)}/{len(weights)}',
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 打印结果
        print(f"\n{title}:")
        print(f"  训练 R²: {train_score:.4f}")
        print(f"  测试 R²: {test_score:.4f}")
        print(f"  非零权重数: {np.sum(np.abs(weights) > 0.01)}/{len(weights)}")
        print(f"  权重范围: [{weights.min():.2f}, {weights.max():.2f}]")

    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=100)
    print("\n📊 正则化对比图已保存到: regularization_comparison.png")
    plt.show()


def lambda_sweep(X_train, y_train, X_test, y_test, degree=10):
    """
    探索正则化强度 λ 的影响
    """
    print("\n" + "=" * 70)
    print("🧪 实验：正则化强度 λ 的影响")
    print("=" * 70)

    X_train_poly = add_polynomial_features(X_train, degree)
    X_test_poly = add_polynomial_features(X_test, degree)

    # 测试不同的 λ 值
    lambdas = np.logspace(-4, 2, 20)  # 0.0001 到 100

    results_l1 = {'train': [], 'test': [], 'non_zero': []}
    results_l2 = {'train': [], 'test': [], 'non_zero': []}

    for lambda_ in lambdas:
        # L1
        model_l1 = RegularizedLinearRegression('l1', lambda_, learning_rate=0.01, n_epochs=1000)
        model_l1.fit(X_train_poly, y_train)
        results_l1['train'].append(model_l1.score(X_train_poly, y_train))
        results_l1['test'].append(model_l1.score(X_test_poly, y_test))
        results_l1['non_zero'].append(np.sum(np.abs(model_l1.weights) > 0.01))

        # L2
        model_l2 = RegularizedLinearRegression('l2', lambda_, learning_rate=0.01, n_epochs=1000)
        model_l2.fit(X_train_poly, y_train)
        results_l2['train'].append(model_l2.score(X_train_poly, y_train))
        results_l2['test'].append(model_l2.score(X_test_poly, y_test))
        results_l2['non_zero'].append(np.sum(np.abs(model_l2.weights) > 0.01))

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² vs λ
    ax1 = axes[0]
    ax1.semilogx(lambdas, results_l1['test'], 'o-', label='L1 Test', linewidth=2, color='blue')
    ax1.semilogx(lambdas, results_l2['test'], 's-', label='L2 Test', linewidth=2, color='red')
    ax1.semilogx(lambdas, results_l1['train'], 'o--', label='L1 Train', linewidth=1,
                 alpha=0.5, color='blue')
    ax1.semilogx(lambdas, results_l2['train'], 's--', label='L2 Train', linewidth=1,
                 alpha=0.5, color='red')
    ax1.set_xlabel('λ (Regularization Strength)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance vs λ', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 非零权重数 vs λ
    ax2 = axes[1]
    ax2.semilogx(lambdas, results_l1['non_zero'], 'o-', label='L1', linewidth=2, color='blue')
    ax2.semilogx(lambdas, results_l2['non_zero'], 's-', label='L2', linewidth=2, color='red')
    ax2.set_xlabel('λ (Regularization Strength)', fontsize=12)
    ax2.set_ylabel('Number of Non-zero Weights', fontsize=12)
    ax2.set_title('Model Sparsity vs λ', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lambda_sweep.png', dpi=100)
    print("\n📊 λ 扫描图已保存到: lambda_sweep.png")
    plt.show()

    print("\n💡 观察：")
    print("  λ 太小 → 几乎无正则化，可能过拟合")
    print("  λ 适中 → 测试 R² 达到峰值（最佳泛化）")
    print("  λ 太大 → 权重被过度压缩，欠拟合")
    print("\n  L1 特性：随着 λ 增大，非零权重数减少（稀疏解）")
    print("  L2 特性：权重变小但不为 0（密集解）")


# ==================== 5. 主程序 ====================
def main():
    print("=" * 70)
    print("正则化 (Regularization) - 防止过拟合")
    print("=" * 70)

    # 1. 演示过拟合
    plot_overfitting_demo()

    # 2. 生成数据
    print("\n" + "=" * 70)
    print("📊 生成训练和测试数据")
    print("=" * 70)
    X_train, y_train = generate_overfitting_data(n_samples=30, noise=3)
    X_test, y_test = generate_overfitting_data(n_samples=100, noise=3, random_state=123)
    print(f"训练集: {len(X_train)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")

    # 3. 对比正则化方法
    compare_regularization(X_train, y_train, X_test, y_test, degree=10)

    # 4. λ 扫描
    lambda_sweep(X_train, y_train, X_test, y_test, degree=10)

    # 5. 总结
    print("\n" + "=" * 70)
    print("✅ 核心要点总结")
    print("=" * 70)
    print("""
1. 过拟合 vs 欠拟合
   - 过拟合：训练好，测试差（模型太复杂）
   - 欠拟合：训练差，测试差（模型太简单）
   - 目标：找到恰当的模型复杂度

2. 正则化的作用
   - 在损失函数中添加对权重的惩罚
   - 防止权重过大 → 模型更平滑
   - 强制模型简化 → 提高泛化能力

3. L1 正则化 (Lasso)
   公式：Loss = MSE + λ * Σ|w|
   特点：
   - 产生稀疏解（很多权重 = 0）
   - 可用于特征选择
   - 梯度：λ * sign(w)

4. L2 正则化 (Ridge)
   公式：Loss = MSE + λ * Σw²
   特点：
   - 权重变小但不为 0
   - 所有特征都保留
   - 也叫"权重衰减"
   - 梯度：2λw

5. 选择 λ (正则化强度)
   - λ = 0：无正则化
   - λ 小：轻微正则化
   - λ 适中：最佳泛化（通过交叉验证找到）
   - λ 大：欠拟合

6. L1 vs L2 如何选择？
   - 特征很多，怀疑很多不重要 → L1（自动特征选择）
   - 所有特征都可能有用 → L2（更稳定）
   - 不确定 → 都试试，或用 Elastic Net（L1+L2）

7. 应用场景
   ✓ 高维数据（特征数 >> 样本数）
   ✓ 多项式特征（容易过拟合）
   ✓ 深度学习（权重衰减是标配）
   ✓ 特征选择（L1）

8. 实践建议
   - 总是标准化特征（正则化对特征尺度敏感）
   - 用交叉验证选择最佳 λ
   - 观察训练/测试曲线判断过拟合
   - 不要正则化偏置项 bias
    """)


# ==================== 6. 与 sklearn 对比 ====================
def sklearn_comparison():
    print("\n" + "=" * 70)
    print("🔬 与 sklearn 对比")
    print("=" * 70)

    from sklearn.linear_model import Ridge, Lasso

    # 生成数据
    X_train, y_train = generate_overfitting_data(n_samples=30, noise=3)
    X_test, y_test = generate_overfitting_data(n_samples=100, noise=3, random_state=123)

    X_train_poly = add_polynomial_features(X_train, 10)
    X_test_poly = add_polynomial_features(X_test, 10)

    # sklearn Ridge
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train_poly, y_train)
    print(f"\nsklearn Ridge:")
    print(f"  训练 R²: {ridge.score(X_train_poly, y_train):.4f}")
    print(f"  测试 R²: {ridge.score(X_test_poly, y_test):.4f}")

    # sklearn Lasso
    lasso = Lasso(alpha=0.1, max_iter=5000)
    lasso.fit(X_train_poly, y_train)
    print(f"\nsklearn Lasso:")
    print(f"  训练 R²: {lasso.score(X_train_poly, y_train):.4f}")
    print(f"  测试 R²: {lasso.score(X_test_poly, y_test):.4f}")
    print(f"  非零权重: {np.sum(np.abs(lasso.coef_) > 0.01)}/{len(lasso.coef_)}")

    print(f"\n✅ sklearn 实现更优化，但原理相同！")


if __name__ == "__main__":
    main()
    sklearn_comparison()

    print("\n💡 练习建议：")
    print("  1. 修改多项式度数（5, 10, 15, 20），观察过拟合程度")
    print("  2. 尝试不同的 λ 值，找到最佳值")
    print("  3. 比较 L1 和 L2 在高维数据上的表现")
    print("  4. 思考：为什么 L1 能产生稀疏解，L2 不能？")
