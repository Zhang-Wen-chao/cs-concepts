"""
神经网络基础 (Neural Networks - Multi-Layer Perceptron)

问题：逻辑回归只能学习线性决策边界，如何处理非线性问题？
目标：通过多层神经网络学习非线性映射

核心概念：
1. 多层结构：输入层 → 隐藏层 → 输出层
2. 非线性激活：ReLU、Sigmoid、Tanh
3. 前向传播：从输入到输出的计算过程
4. 反向传播：从输出到输入的梯度计算
5. 通用近似定理：单隐藏层神经网络可以逼近任意连续函数

从逻辑回归到神经网络：
- 逻辑回归 = 单层神经网络（线性 + Sigmoid）
- 神经网络 = 多层 + 非线性激活（可学习非线性关系）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==================== 1. 激活函数 ====================
class ActivationFunctions:
    """
    激活函数集合

    ====================================================================
    🔑 为什么需要激活函数？
    ====================================================================

    如果没有激活函数（或只用线性激活）：
    多层网络 = 单层网络

    推导：
    y = W2(W1x + b1) + b2
      = W2W1x + W2b1 + b2
      = W'x + b'  ← 还是线性的！

    结论：无论多少层，没有非线性激活，网络只能学习线性关系

    激活函数的作用：
    1. 引入非线性 → 网络可以逼近任意函数
    2. 打破对称性 → 不同神经元学习不同特征
    3. 控制输出范围 → 数值稳定

    ====================================================================
    """

    @staticmethod
    def relu(z):
        """
        ReLU (Rectified Linear Unit)
        f(z) = max(0, z)

        优点：
        - 计算简单
        - 缓解梯度消失（正区间梯度=1）
        - 稀疏激活（负值为0）
        - 深度学习最常用

        缺点：
        - 负区间梯度=0（Dead ReLU）
        - 输出无上界
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """ReLU 导数：1 if z>0, else 0"""
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid
        f(z) = 1 / (1 + e^(-z))

        优点：
        - 输出 (0, 1)，可解释为概率
        - 平滑连续

        缺点：
        - 梯度消失（两端梯度接近0）
        - 输出非零中心（影响优化）
        - 计算 exp 较慢
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def sigmoid_derivative(z):
        """Sigmoid 导数：σ(z) * (1 - σ(z))"""
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        """
        Tanh (Hyperbolic Tangent)
        f(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        优点：
        - 输出 (-1, 1)，零中心
        - 比 Sigmoid 好

        缺点：
        - 仍有梯度消失
        - 计算 exp 较慢
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        """Tanh 导数：1 - tanh²(z)"""
        t = np.tanh(z)
        return 1 - t ** 2


# ==================== 2. 神经网络类 ====================
class NeuralNetwork:
    """
    两层神经网络（1个隐藏层）

    结构：
    输入层 (n_features) → 隐藏层 (n_hidden) → 输出层 (n_outputs)
    """

    def __init__(self, n_features, n_hidden, n_outputs,
                 activation='relu', learning_rate=0.01, n_epochs=1000, batch_size=32):
        """
        参数：
            n_features: 输入特征数
            n_hidden: 隐藏层神经元数
            n_outputs: 输出数（分类类别数）
            activation: 隐藏层激活函数 ('relu', 'sigmoid', 'tanh')
            learning_rate: 学习率
            n_epochs: 训练轮数
            batch_size: 批量大小
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # 选择激活函数
        self.activation_name = activation
        if activation == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif activation == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 初始化参数
        self._initialize_weights()

        self.loss_history = []

    def _initialize_weights(self):
        """
        参数初始化

        ====================================================================
        🔑 为什么需要随机初始化？
        ====================================================================

        如果所有权重初始化为 0：
        - 所有神经元计算相同的结果
        - 所有梯度相同
        - 无法打破对称性 → 网络退化

        常用初始化方法：
        1. Xavier 初始化（tanh/sigmoid）
           W ~ N(0, 1/sqrt(n_in))

        2. He 初始化（ReLU）
           W ~ N(0, 2/sqrt(n_in))

        3. 小随机数
           W ~ N(0, 0.01)

        ====================================================================
        """
        # He 初始化（适合 ReLU）
        self.W1 = np.random.randn(self.n_features, self.n_hidden) * np.sqrt(2.0 / self.n_features)
        self.b1 = np.zeros((1, self.n_hidden))

        self.W2 = np.random.randn(self.n_hidden, self.n_outputs) * np.sqrt(2.0 / self.n_hidden)
        self.b2 = np.zeros((1, self.n_outputs))

    def forward(self, X):
        """
        前向传播

        ====================================================================
        🔑 前向传播流程
        ====================================================================

        层1（输入 → 隐藏）：
        Z1 = X @ W1 + b1           ← 线性变换
        A1 = activation(Z1)        ← 非线性激活

        层2（隐藏 → 输出）：
        Z2 = A1 @ W2 + b2          ← 线性变换
        A2 = softmax(Z2)           ← Softmax（多分类）

        符号说明：
        - Z: 线性输出（未激活）
        - A: 激活后的输出
        - @: 矩阵乘法

        ====================================================================
        """
        # 层1：输入 → 隐藏
        self.Z1 = np.dot(X, self.W1) + self.b1  # (batch_size, n_hidden)
        self.A1 = self.activation(self.Z1)      # (batch_size, n_hidden)

        # 层2：隐藏 → 输出
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # (batch_size, n_outputs)
        self.A2 = self._softmax(self.Z2)              # (batch_size, n_outputs)

        return self.A2

    def backward(self, X, y_true, y_pred):
        """
        反向传播

        ====================================================================
        🔑 反向传播推导（链式法则）
        ====================================================================

        损失函数：L = CrossEntropy(y_true, y_pred)

        目标：计算 ∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2

        【层2的梯度（输出层）】

        1. 输出层误差
           dZ2 = ∂L/∂Z2 = y_pred - y_true

           这是交叉熵 + Softmax 的神奇简化！
           （推导过程类似逻辑回归）

        2. W2 的梯度
           ∂L/∂W2 = ∂L/∂Z2 · ∂Z2/∂W2
                  = A1^T @ dZ2

        3. b2 的梯度
           ∂L/∂b2 = sum(dZ2, axis=0)

        【层1的梯度（隐藏层）】

        4. 隐藏层误差（通过层2反传）
           dA1 = dZ2 @ W2^T
           dZ1 = dA1 * activation'(Z1)

        5. W1 的梯度
           ∂L/∂W1 = X^T @ dZ1

        6. b1 的梯度
           ∂L/∂b1 = sum(dZ1, axis=0)

        核心思想：
        - 从输出层开始，逐层反向计算梯度
        - 使用链式法则连接各层
        - 误差通过权重矩阵反向传播

        ====================================================================
        """
        m = X.shape[0]

        # ========== 层2梯度（输出层） ==========
        dZ2 = y_pred - y_true  # (batch_size, n_outputs)

        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)  # (n_hidden, n_outputs)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # (1, n_outputs)

        # ========== 层1梯度（隐藏层） ==========
        # 误差反向传播到隐藏层
        dA1 = np.dot(dZ2, self.W2.T)  # (batch_size, n_hidden)

        # 乘以激活函数的导数
        dZ1 = dA1 * self.activation_derivative(self.Z1)  # (batch_size, n_hidden)

        dW1 = (1 / m) * np.dot(X.T, dZ1)  # (n_features, n_hidden)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)  # (1, n_hidden)

        return dW1, db1, dW2, db2

    def fit(self, X, y):
        """
        训练模型

        流程：
        1. 前向传播：计算预测值
        2. 计算损失
        3. 反向传播：计算梯度
        4. 更新参数
        """
        n_samples = X.shape[0]

        # 转换为 one-hot 编码
        y_one_hot = self._to_one_hot(y)

        # Mini-batch 梯度下降
        for epoch in range(self.n_epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)

            # 分批训练
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_one_hot[batch_indices]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 反向传播
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch, y_pred)

                # 更新参数
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            # 记录损失
            if epoch % 10 == 0:
                y_pred_all = self.forward(X)
                loss = self._cross_entropy_loss(y_one_hot, y_pred_all)
                self.loss_history.append(loss)

    def predict_proba(self, X):
        """预测概率"""
        return self.forward(X)

    def predict(self, X):
        """预测类别"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    @staticmethod
    def _softmax(z):
        """Softmax 激活函数"""
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def _cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
        """交叉熵损失"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def _to_one_hot(self, y):
        """转换为 one-hot 编码"""
        one_hot = np.zeros((y.shape[0], self.n_outputs))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot


# ==================== 3. 数据生成 ====================
def generate_nonlinear_data(dataset='moons', n_samples=300, noise=0.2, random_state=42):
    """
    生成非线性数据

    dataset: 'moons', 'circles', 'linear'
    """
    if dataset == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif dataset == 'linear':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1,
                                   random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return X, y


# ==================== 4. 可视化 ====================
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """绘制决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu', levels=1)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', marker='o',
                s=50, edgecolors='k', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', marker='s',
                s=50, edgecolors='k', label='Class 1', alpha=0.7)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)


def compare_with_logistic_regression():
    """
    对比神经网络和逻辑回归

    在非线性数据上，神经网络明显优于逻辑回归
    """
    print("=" * 70)
    print("🔬 对比：神经网络 vs 逻辑回归（非线性数据）")
    print("=" * 70)

    # 生成非线性数据
    X, y = generate_nonlinear_data('moons', n_samples=300, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 1. 逻辑回归（线性模型）
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_train_acc = lr.score(X_train, y_train)
    lr_test_acc = lr.score(X_test, y_test)

    # 2. 神经网络
    nn = NeuralNetwork(n_features=2, n_hidden=10, n_outputs=2,
                      activation='relu', learning_rate=0.1, n_epochs=500)
    nn.fit(X_train, y_train)
    nn_train_acc = nn.score(X_train, y_train)
    nn_test_acc = nn.score(X_test, y_test)

    print(f"\n逻辑回归（线性模型）：")
    print(f"  训练准确率: {lr_train_acc:.4f} ({lr_train_acc*100:.2f}%)")
    print(f"  测试准确率: {lr_test_acc:.4f} ({lr_test_acc*100:.2f}%)")

    print(f"\n神经网络（非线性模型）：")
    print(f"  训练准确率: {nn_train_acc:.4f} ({nn_train_acc*100:.2f}%)")
    print(f"  测试准确率: {nn_test_acc:.4f} ({nn_test_acc*100:.2f}%)")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plot_decision_boundary(lr, X_test, y_test,
                          title=f'Logistic Regression (Linear)\nTest Acc: {lr_test_acc:.2%}')

    plt.subplot(1, 2, 2)
    plot_decision_boundary(nn, X_test, y_test,
                          title=f'Neural Network (Non-linear)\nTest Acc: {nn_test_acc:.2%}')

    plt.tight_layout()
    plt.savefig('nn_vs_lr.png', dpi=100)
    print("\n📊 对比图已保存到: nn_vs_lr.png")
    plt.show()

    print("\n💡 观察：")
    print("  - 逻辑回归只能学习线性边界 → 在非线性数据上表现差")
    print("  - 神经网络可以学习复杂的非线性边界 → 准确率显著提升")


def compare_activations():
    """对比不同激活函数"""
    print("\n" + "=" * 70)
    print("🧪 实验：不同激活函数的影响")
    print("=" * 70)

    X, y = generate_nonlinear_data('moons', n_samples=300, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    activations = ['relu', 'sigmoid', 'tanh']
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, activation in enumerate(activations):
        print(f"\n训练 {activation.upper()} 激活...")

        model = NeuralNetwork(n_features=2, n_hidden=10, n_outputs=2,
                             activation=activation, learning_rate=0.1, n_epochs=500)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        results[activation] = {'train': train_acc, 'test': test_acc}

        print(f"  测试准确率: {test_acc:.4f}")

        # 可视化
        plt.subplot(1, 3, idx + 1)
        plot_decision_boundary(model, X_test, y_test,
                              title=f'{activation.upper()} Activation\nTest Acc: {test_acc:.2%}')

    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=100)
    print("\n📊 激活函数对比图已保存到: activation_comparison.png")
    plt.show()

    print("\n💡 观察：")
    print("  - ReLU: 最常用，训练快，性能好")
    print("  - Sigmoid: 容易梯度消失，性能稍差")
    print("  - Tanh: 比 Sigmoid 好，但不如 ReLU")


def visualize_hidden_neurons():
    """可视化隐藏层神经元数量的影响"""
    print("\n" + "=" * 70)
    print("🧪 实验：隐藏层神经元数量的影响")
    print("=" * 70)

    X, y = generate_nonlinear_data('circles', n_samples=300, noise=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_hidden_list = [2, 5, 10, 20]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, n_hidden in enumerate(n_hidden_list):
        print(f"\n隐藏层神经元数: {n_hidden}")

        model = NeuralNetwork(n_features=2, n_hidden=n_hidden, n_outputs=2,
                             activation='relu', learning_rate=0.1, n_epochs=500)
        model.fit(X_train, y_train)

        test_acc = model.score(X_test, y_test)
        print(f"  测试准确率: {test_acc:.4f}")

        plt.subplot(2, 2, idx + 1)
        plot_decision_boundary(model, X_test, y_test,
                              title=f'Hidden Units: {n_hidden}\nTest Acc: {test_acc:.2%}')

    plt.tight_layout()
    plt.savefig('hidden_neurons.png', dpi=100)
    print("\n📊 隐藏神经元对比图已保存到: hidden_neurons.png")
    plt.show()

    print("\n💡 观察：")
    print("  - 神经元太少 → 容量不足，无法学习复杂模式")
    print("  - 神经元适中 → 恰到好处")
    print("  - 神经元太多 → 可能过拟合（需要正则化）")


# ==================== 5. 主程序 ====================
def main():
    print("=" * 70)
    print("神经网络基础 (Neural Networks / MLP)")
    print("=" * 70)

    # 1. 神经网络 vs 逻辑回归
    compare_with_logistic_regression()

    # 2. 对比激活函数
    compare_activations()

    # 3. 隐藏层大小的影响
    visualize_hidden_neurons()

    # 4. 总结
    print("\n" + "=" * 70)
    print("✅ 核心要点总结")
    print("=" * 70)
    print("""
1. 神经网络 = 多层 + 非线性激活
   - 单层（逻辑回归）→ 只能学习线性
   - 多层 + 激活 → 可以学习非线性

2. 网络结构
   输入层 → [隐藏层1 → ... → 隐藏层N] → 输出层
   - 隐藏层：提取特征，学习表示
   - 输出层：最终决策

3. 前向传播
   逐层计算：线性变换 + 非线性激活
   Z = W·A_prev + b
   A = activation(Z)

4. 反向传播（核心算法）
   从输出层开始，逐层反向计算梯度
   - 使用链式法则
   - 误差通过权重矩阵反传
   - dZ_l = dA_l * activation'(Z_l)

5. 激活函数选择
   - ReLU: 默认选择（深度学习标配）
   - Sigmoid: 输出层（二分类）
   - Tanh: 比 Sigmoid 好，但不如 ReLU
   - Softmax: 输出层（多分类）

6. 参数初始化
   - 不能全 0（打破对称性）
   - He 初始化（ReLU）
   - Xavier 初始化（Tanh/Sigmoid）

7. 隐藏层大小
   - 太小：容量不足
   - 适中：恰到好处
   - 太大：可能过拟合

8. 通用近似定理
   单隐藏层神经网络可以逼近任意连续函数
   （前提：隐藏层足够大）

9. 与深度学习的关系
   - MLP 是深度学习的基础
   - CNN、RNN、Transformer 都基于相同原理
   - 只是网络结构更复杂

10. 实践要点
    ✓ 标准化输入数据
    ✓ 使用 ReLU 激活
    ✓ He 初始化
    ✓ Mini-batch 梯度下降
    ✓ 监控训练/测试曲线（防止过拟合）
    """)


if __name__ == "__main__":
    main()

    print("\n💡 练习建议：")
    print("  1. 尝试 3 层网络（2个隐藏层），观察性能")
    print("  2. 在不同数据集上测试（moons、circles）")
    print("  3. 手动推导一遍反向传播公式（非常重要！）")
    print("  4. 思考：为什么深度网络比浅层网络更强大？")
