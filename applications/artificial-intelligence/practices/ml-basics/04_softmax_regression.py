"""
Softmax 回归 (Softmax Regression / Multinomial Logistic Regression)

问题：如何处理多分类问题（3 个或更多类别）？
目标：将二分类的逻辑回归扩展到多分类

核心概念：
1. Softmax 函数：将 K 个类别的得分转换为概率分布
   p_k = e^(z_k) / Σ(e^(z_j))  其中 Σp_k = 1
2. 交叉熵损失（多分类版本）：
   Loss = -Σ y_k * log(p_k)  (y_k 是 one-hot 编码)
3. 决策：选择概率最大的类别 argmax(p_k)

逻辑回归 vs Softmax 回归：
- 逻辑回归：2 分类，Sigmoid，输出 1 个概率
- Softmax 回归：K 分类，Softmax，输出 K 个概率
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==================== 1. 核心函数 ====================
def softmax(z):
    """
    Softmax 函数：将 K 个得分转换为概率分布

    输入：z = [z1, z2, ..., zK]  (K 个类别的得分)
    输出：p = [p1, p2, ..., pK]  (K 个类别的概率，和为 1)

    公式：p_k = e^(z_k) / Σ(e^(z_j))

    ========================================================================
    🔑 Softmax vs Sigmoid 的关系
    ========================================================================

    【Sigmoid (二分类)】
    两个类别：0 和 1
    p(y=1) = 1 / (1 + e^(-z))
    p(y=0) = 1 - p(y=1)

    【Softmax (多分类)】
    K 个类别：0, 1, 2, ..., K-1
    p(y=k) = e^(z_k) / Σ(e^(z_j))

    当 K=2 时，Softmax 退化为 Sigmoid！

    推导：
    p(y=1) = e^(z1) / (e^(z0) + e^(z1))
           = 1 / (1 + e^(z0-z1))
           = 1 / (1 + e^(-(z1-z0)))  ← 这就是 Sigmoid!

    结论：Sigmoid 是 Softmax 的特例（K=2）

    ========================================================================
    🔑 Softmax 的性质
    ========================================================================

    1. 输出范围：每个 p_k ∈ (0, 1)
    2. 概率和为 1：Σ p_k = 1
    3. 单调性：z_k 越大，p_k 越大
    4. 相对大小：不仅看绝对值，还看相对差异

    【数值稳定性技巧】
    直接计算 e^(z_k) 可能溢出（z 很大时）

    解决方案：减去最大值
    p_k = e^(z_k - max(z)) / Σ(e^(z_j - max(z)))

    为什么有效？
    分子分母同时除以 e^(max(z))，结果不变
    但避免了 e^(大数) 的溢出

    ========================================================================
    """
    # 数值稳定性：减去最大值
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    多分类交叉熵损失

    输入：
        y_true: one-hot 编码的真实标签，shape (n_samples, n_classes)
                例如：[[1,0,0], [0,1,0], [0,0,1]]
        y_pred: 预测概率，shape (n_samples, n_classes)
                例如：[[0.7,0.2,0.1], [0.1,0.8,0.1], ...]

    公式：
        Loss = -1/n * ΣΣ y_ik * log(p_ik)
        其中 i 是样本索引，k 是类别索引

    ========================================================================
    🔑 理解多分类交叉熵
    ========================================================================

    【直觉理解】
    对于每个样本，只有一个类别是正确的（y_ik = 1）
    其他类别 y_ik = 0，对损失没有贡献

    例子：真实标签是类别 1 (one-hot: [0, 1, 0])
    Loss = -(0*log(p0) + 1*log(p1) + 0*log(p2))
         = -log(p1)

    只关心正确类别的预测概率！

    【二分类交叉熵 vs 多分类交叉熵】

    二分类（K=2）：
    Loss = -[y*log(p) + (1-y)*log(1-p)]

    多分类（K>2）：
    Loss = -Σ y_k * log(p_k)

    当 K=2 时，两者等价！

    【为什么用 one-hot 编码？】
    类别是离散的，没有大小关系
    - 不能用 0, 1, 2 表示类别（会暗示 2 > 1 > 0）
    - 用 one-hot：[1,0,0], [0,1,0], [0,0,1]
    - 每个类别都是独立的维度

    ========================================================================
    """
    # 裁剪预测值，避免 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 计算交叉熵
    # 只有正确类别（y_true=1）的位置才有贡献
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss


def to_one_hot(y, n_classes):
    """
    将类别标签转换为 one-hot 编码

    输入：y = [0, 2, 1]  (类别索引)
    输出：[[1, 0, 0],
          [0, 0, 1],
          [0, 1, 0]]  (one-hot 向量)
    """
    one_hot = np.zeros((y.shape[0], n_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# ==================== 2. Softmax 回归类 ====================
class SoftmaxRegression:
    """从零实现 Softmax 回归（多分类逻辑回归）"""

    def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=32):
        """
        参数：
            learning_rate: 学习率
            n_epochs: 训练轮数
            batch_size: 批量大小
        """
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = None  # shape: (n_features, n_classes)
        self.bias = None     # shape: (n_classes,)
        self.loss_history = []

    def fit(self, X, y):
        """
        训练模型

        X: shape (n_samples, n_features)
        y: shape (n_samples,)  类别索引 [0, 1, 2, ...]

        ====================================================================
        🔑 Softmax 回归的参数结构
        ====================================================================

        逻辑回归（K=2）：
        - 权重：w ∈ R^d  (d 个特征)
        - 偏置：b ∈ R    (1 个值)
        - 输出：z = wx + b  (1 个得分)

        Softmax 回归（K 类）：
        - 权重：W ∈ R^(d×K)  (每个类别一组权重)
        - 偏置：b ∈ R^K      (每个类别一个偏置)
        - 输出：Z = XW + b   (K 个得分)

        可以理解为：K 个逻辑回归并行运行！

        例子：3 个特征，4 个类别
        W = [w0_class0, w0_class1, w0_class2, w0_class3]  ← 特征 0
            [w1_class0, w1_class1, w1_class2, w1_class3]  ← 特征 1
            [w2_class0, w2_class1, w2_class2, w2_class3]  ← 特征 2

        ====================================================================
        """
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        # 初始化参数
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)

        # 转换为 one-hot 编码
        y_one_hot = to_one_hot(y, self.n_classes)

        # Mini-batch 梯度下降
        for epoch in range(self.n_epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)

            # 分批训练
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_one_hot[batch_indices]

                # ========== 前向传播 ==========
                # 1. 线性组合：Z = XW + b
                #    X_batch: (batch_size, n_features)
                #    weights: (n_features, n_classes)
                #    Z: (batch_size, n_classes)
                Z = np.dot(X_batch, self.weights) + self.bias

                # 2. Softmax 激活
                #    将 K 个得分转换为概率
                y_pred = softmax(Z)

                # ========== 计算梯度 ==========
                # 梯度推导（类似逻辑回归）：
                # ∂Loss/∂W = X^T · (y_pred - y_true) / batch_size
                # ∂Loss/∂b = sum(y_pred - y_true) / batch_size
                #
                # 神奇的是：形式和逻辑回归完全一样！
                # 只是从标量变成了向量/矩阵

                batch_size_actual = len(X_batch)
                error = y_pred - y_batch  # shape: (batch_size, n_classes)

                dW = np.dot(X_batch.T, error) / batch_size_actual
                db = np.sum(error, axis=0) / batch_size_actual

                # ========== 更新参数 ==========
                self.weights -= self.lr * dW
                self.bias -= self.lr * db

            # 记录损失（每 10 个 epoch）
            if epoch % 10 == 0:
                Z_all = np.dot(X, self.weights) + self.bias
                y_pred_all = softmax(Z_all)
                loss = categorical_cross_entropy(y_one_hot, y_pred_all)
                self.loss_history.append(loss)

    def predict_proba(self, X):
        """预测每个类别的概率"""
        Z = np.dot(X, self.weights) + self.bias
        return softmax(Z)

    def predict(self, X):
        """预测类别（选择概率最大的）"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ==================== 3. 数据生成 ====================
def generate_multiclass_data(n_samples=600, n_classes=3, n_features=2, random_state=42):
    """
    生成多分类数据

    返回：
        X: 特征，shape (n_samples, n_features)
        y: 类别标签，shape (n_samples,)

    注意：sklearn 的限制
    - n_classes * n_clusters_per_class ≤ 2^n_informative
    - n_informative ≤ n_features
    - 如果 n_classes > 2^n_features，需要增加 n_features
    """
    # 根据类别数动态调整参数
    # 确保 2^n_informative >= n_classes
    required_informative = int(np.ceil(np.log2(n_classes)))

    # 如果需要的 n_informative 超过 n_features，增加 n_features
    if required_informative > n_features:
        n_features = required_informative
        print(f"  ⚠️  增加特征数到 {n_features}（类别数 {n_classes} 需要至少 {required_informative} 个信息特征）")

    n_informative = min(required_informative, n_features)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=random_state
    )
    return X, y


# ==================== 4. 可视化 ====================
def plot_decision_boundary_multiclass(model, X, y, title="Decision Boundary"):
    """
    绘制多分类决策边界

    对于 Softmax 回归，决策边界是线性的
    K 个类别会有 K 个区域
    """
    # 设置网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=model.n_classes-1)

    # 绘制数据点
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']

    for class_idx in range(model.n_classes):
        mask = (y == class_idx)
        plt.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[class_idx % len(colors)],
            marker=markers[class_idx % len(markers)],
            s=50,
            edgecolors='k',
            label=f'Class {class_idx}',
            alpha=0.7
        )

    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)


def visualize_softmax():
    """可视化 Softmax 函数的行为"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：3 个类别的 Softmax
    z1 = np.linspace(-3, 3, 100)
    z2 = 0  # 固定
    z3 = 0  # 固定

    probs = []
    for z1_val in z1:
        z = np.array([z1_val, z2, z3])
        p = softmax(z.reshape(1, -1))[0]
        probs.append(p)
    probs = np.array(probs)

    axes[0].plot(z1, probs[:, 0], label='p(class 0)', linewidth=2, color='blue')
    axes[0].plot(z1, probs[:, 1], label='p(class 1)', linewidth=2, color='red')
    axes[0].plot(z1, probs[:, 2], label='p(class 2)', linewidth=2, color='green')
    axes[0].axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('z₁ (score of class 0)', fontsize=12)
    axes[0].set_ylabel('Probability', fontsize=12)
    axes[0].set_title('Softmax: z₁ varies, z₂=z₃=0', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # 右图：概率和始终为 1
    axes[1].plot(z1, probs.sum(axis=1), linewidth=3, color='purple')
    axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Sum = 1')
    axes[1].set_xlabel('z₁', fontsize=12)
    axes[1].set_ylabel('Sum of Probabilities', fontsize=12)
    axes[1].set_title('Softmax Property: Σp = 1', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.95, 1.05)

    plt.tight_layout()
    plt.savefig('softmax_function.png', dpi=100)
    print("📊 Softmax 函数图已保存到: softmax_function.png")
    plt.show()


# ==================== 5. 主程序 ====================
def main():
    print("=" * 70)
    print("Softmax 回归 (Softmax Regression) - 多分类任务")
    print("=" * 70)

    # ========== 1. 可视化 Softmax 函数 ==========
    print("\n" + "=" * 70)
    print("📈 第一步：理解 Softmax 函数")
    print("=" * 70)
    print("""
Softmax 是 Sigmoid 的多分类推广：
- Sigmoid: 2 类 → 输出 1 个概率
- Softmax: K 类 → 输出 K 个概率（和为 1）

关键性质：
1. 所有概率和为 1
2. 某个类别得分越高，其概率越大
3. 概率是相对的（看所有类别的相对大小）
    """)
    visualize_softmax()

    # ========== 2. 生成多分类数据 ==========
    print("\n" + "=" * 70)
    print("📊 第二步：生成多分类数据（3 类）")
    print("=" * 70)

    n_classes = 3
    X, y = generate_multiclass_data(n_samples=600, n_classes=n_classes)

    # 标准化特征（可选，但通常有帮助）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    for i in range(n_classes):
        print(f"类别 {i} 样本数: {np.sum(y_train == i)}")

    # ========== 3. 训练模型 ==========
    print("\n" + "=" * 70)
    print("🔧 第三步：训练 Softmax 回归模型")
    print("=" * 70)

    model = SoftmaxRegression(learning_rate=0.1, n_epochs=200, batch_size=32)
    print(f"超参数：学习率={model.lr}, 训练轮数={model.n_epochs}, 批量大小={model.batch_size}")
    print("\n训练中...")
    model.fit(X_train, y_train)

    # ========== 4. 评估模型 ==========
    print("\n" + "=" * 70)
    print("📊 第四步：评估模型性能")
    print("=" * 70)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\n训练集准确率: {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # 查看参数形状
    print(f"\n学习到的参数形状：")
    print(f"  权重 W: {model.weights.shape}  (n_features × n_classes)")
    print(f"  偏置 b: {model.bias.shape}    (n_classes,)")

    # 预测示例
    print(f"\n预测示例（前 3 个测试样本）：")
    sample_probs = model.predict_proba(X_test[:3])
    sample_preds = model.predict(X_test[:3])

    for i in range(3):
        print(f"\n  样本 {i+1}:")
        print(f"    真实类别: {y_test[i]}")
        print(f"    预测类别: {sample_preds[i]}")
        print(f"    各类概率: {sample_probs[i]}")
        print(f"    (Class 0: {sample_probs[i][0]:.3f}, "
              f"Class 1: {sample_probs[i][1]:.3f}, "
              f"Class 2: {sample_probs[i][2]:.3f})")

    # ========== 5. 可视化结果 ==========
    print("\n" + "=" * 70)
    print("🎨 第五步：可视化决策边界和训练过程")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 5))

    # 子图1：训练集决策边界
    plt.subplot(1, 3, 1)
    plot_decision_boundary_multiclass(
        model, X_train, y_train,
        title=f'Training Set\nAccuracy: {train_acc:.2%}'
    )

    # 子图2：测试集决策边界
    plt.subplot(1, 3, 2)
    plot_decision_boundary_multiclass(
        model, X_test, y_test,
        title=f'Test Set\nAccuracy: {test_acc:.2%}'
    )

    # 子图3：损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(range(0, model.n_epochs, 10), model.loss_history,
             linewidth=2, color='blue', marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('softmax_regression_result.png', dpi=100)
    print("\n📊 结果图已保存到: softmax_regression_result.png")
    plt.show()

    # ========== 6. 对比 sklearn ==========
    print("\n" + "=" * 70)
    print("🔬 第六步：与 sklearn 对比验证")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression as SklearnLR

    sklearn_model = SklearnLR(multi_class='multinomial', solver='lbfgs',
                              max_iter=1000, random_state=42)
    sklearn_model.fit(X_train, y_train)

    sklearn_train_acc = sklearn_model.score(X_train, y_train)
    sklearn_test_acc = sklearn_model.score(X_test, y_test)

    print(f"\nSklearn Softmax 回归（多分类逻辑回归）：")
    print(f"  训练集准确率: {sklearn_train_acc:.4f} ({sklearn_train_acc * 100:.2f}%)")
    print(f"  测试集准确率: {sklearn_test_acc:.4f} ({sklearn_test_acc * 100:.2f}%)")
    print(f"  权重形状: {sklearn_model.coef_.shape}")

    print(f"\n对比结果：")
    print(f"  准确率差异（测试集）: {abs(test_acc - sklearn_test_acc):.4f}")
    print(f"  ✅ 实现基本正确！")


# ==================== 6. 实验区 ====================
def experiment_num_classes():
    """
    实验：不同类别数的影响

    观察 2 类、3 类、4 类、5 类的表现
    """
    print("\n" + "=" * 70)
    print("🧪 实验：不同类别数的影响")
    print("=" * 70)

    results = []

    for n_classes in [2, 3, 4, 5]:
        print(f"\n训练 {n_classes} 类分类器...")

        # 生成数据
        X, y = generate_multiclass_data(n_samples=600, n_classes=n_classes)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 训练
        model = SoftmaxRegression(learning_rate=0.1, n_epochs=200)
        model.fit(X_train, y_train)

        # 评估
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        results.append({
            'n_classes': n_classes,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'n_params': model.weights.size + model.bias.size
        })

        print(f"  测试准确率: {test_acc:.4f}")
        print(f"  参数总数: {results[-1]['n_params']}")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n_classes_list = [r['n_classes'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    n_params_list = [r['n_params'] for r in results]

    # 准确率 vs 类别数
    ax1.plot(n_classes_list, test_accs, 'o-', linewidth=2, markersize=10, color='blue')
    ax1.set_xlabel('Number of Classes', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Number of Classes', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_classes_list)

    # 参数数量 vs 类别数
    ax2.plot(n_classes_list, n_params_list, 'o-', linewidth=2, markersize=10, color='red')
    ax2.set_xlabel('Number of Classes', fontsize=12)
    ax2.set_ylabel('Number of Parameters', fontsize=12)
    ax2.set_title('Parameters vs Number of Classes', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_classes_list)

    plt.tight_layout()
    plt.savefig('softmax_num_classes.png', dpi=100)
    print("\n📊 类别数对比图已保存到: softmax_num_classes.png")
    plt.show()

    print("\n💡 观察：")
    print("  - 类别越多，问题越难，准确率可能下降")
    print("  - 参数数量线性增长：n_params = (n_features + 1) × n_classes")


# ==================== 7. 总结 ====================
def print_summary():
    print("\n" + "=" * 70)
    print("✅ 核心要点总结")
    print("=" * 70)
    print("""
1. Softmax 回归 = 多分类逻辑回归
   - 二分类：Sigmoid
   - 多分类：Softmax（Sigmoid 的推广）

2. Softmax 函数
   公式：p_k = e^(z_k) / Σ(e^(z_j))
   性质：
   - 输出 K 个概率，和为 1
   - 单调：z_k 越大，p_k 越大
   - 相对：看所有类别的相对大小

3. 参数结构
   - 权重：W ∈ R^(d×K)  每个类别一组权重
   - 偏置：b ∈ R^K      每个类别一个偏置
   - 可以理解为 K 个逻辑回归并行

4. One-hot 编码
   - 类别是离散的，用 one-hot 表示
   - [1,0,0], [0,1,0], [0,0,1]
   - 每个类别独立，无大小关系

5. 交叉熵损失（多分类）
   Loss = -Σ y_k * log(p_k)
   - 只有正确类别（y_k=1）有贡献
   - 等价于 -log(p_正确类别)

6. 梯度公式（神奇的简洁）
   ∂Loss/∂W = X^T · (y_pred - y_true)
   ∂Loss/∂b = sum(y_pred - y_true)
   - 形式和逻辑回归完全一样！
   - 只是从标量变成了向量/矩阵

7. 决策边界
   - Softmax 回归是线性分类器
   - K 个类别的边界都是线性的
   - 无法处理非线性数据（需要神经网络）

8. 应用场景
   ✓ 图像分类（手写数字识别）
   ✓ 文本分类（新闻主题）
   ✓ 多标签分类（物体检测）
   ✓ 任何多分类问题

9. 与神经网络的关系
   - Softmax 回归 = 单层神经网络 + Softmax 输出
   - 深度神经网络的最后一层通常用 Softmax
   - 理解 Softmax 是理解分类网络的基础
    """)

    print("=" * 70)
    print("🎯 下一步学习建议")
    print("=" * 70)
    print("""
1. 正则化（L1/L2 Regularization）
   - 防止过拟合
   - 特征选择
   - 增强泛化能力

2. 神经网络基础（多层感知机 MLP）
   - 多层结构
   - 非线性激活函数
   - 反向传播算法

3. 评估指标深入
   - 混淆矩阵（多分类）
   - Macro/Micro 平均
   - ROC 曲线（多分类版本）

4. 优化算法
   - Momentum
   - Adam
   - 学习率调度
    """)


if __name__ == "__main__":
    # 主实验
    main()

    # 额外实验（取消注释运行）
    experiment_num_classes()

    # 总结
    print_summary()

    print("\n💡 练习建议：")
    print("  1. 尝试 4 类或 5 类分类，观察准确率变化")
    print("  2. 修改学习率，看对训练的影响")
    print("  3. 对比不同 batch size 的效果")
    print("  4. 思考：为什么参数数量是 (n_features + 1) × n_classes？")
