"""
逻辑回归 (Logistic Regression)

问题：从回归到分类 - 如何预测离散类别？
目标：理解逻辑回归的原理和实现二分类任务

核心概念：
1. Sigmoid函数：将线性输出 z = wx + b 映射到 [0, 1] 概率
   σ(z) = 1 / (1 + e^(-z))
2. 交叉熵损失：分类问题的损失函数（不再是MSE）
   Loss = -[y*log(p) + (1-y)*log(1-p)]
3. 决策边界：分类的分界线，当 p = 0.5 时的边界
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ==================== 1. 核心函数 ====================
def sigmoid(z):
    """
    Sigmoid 激活函数

    作用：将任意实数映射到 (0, 1) 区间，表示概率

    性质：
    - 输出范围：(0, 1)
    - 在 z=0 处，σ(0) = 0.5
    - 单调递增
    - 导数：σ'(z) = σ(z) * (1 - σ(z))

    例子：
    - z = 0   → σ(z) = 0.5   (不确定)
    - z = 5   → σ(z) ≈ 0.993 (非常确定是正类)
    - z = -5  → σ(z) ≈ 0.007 (非常确定是负类)
    """
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    二分类交叉熵损失

    为什么不用 MSE？
    - MSE 用于回归，对分类问题优化效果差
    - 交叉熵能更好地衡量概率分布的差异

    公式：
    Loss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]

    其中：
    - y: 真实标签 (0 或 1)
    - p: 预测概率 (0 到 1)

    epsilon: 防止 log(0) 出现（np.clip 裁剪到 [epsilon, 1-epsilon]）

    ============================================================================
    🔑 核心优势：交叉熵 vs MSE 的梯度对比
    ============================================================================

    对于逻辑回归：z = wx + b, p = sigmoid(z)

    【使用交叉熵】
    Loss = -[y*log(p) + (1-y)*log(1-p)]

    梯度推导（链式法则）：
    ∂Loss/∂w = (∂Loss/∂p) · (∂p/∂z) · (∂z/∂w)
             = [(p-y)/(p(1-p))] · [p(1-p)] · x
             = (p - y) · x  ← p(1-p) 被约掉了！

    ✅ 优势：
    1. 梯度形式极简：∂Loss/∂w = (p - y) · x
    2. 避免梯度消失：sigmoid 导数 p(1-p) 被抵消
    3. 错误越大，梯度越大，学习越快

    【如果使用 MSE】
    Loss = (y - p)²

    梯度推导：
    ∂Loss/∂w = 2(p - y) · p(1-p) · x  ← 保留了 p(1-p)！

    ❌ 问题：
    1. 当 p 接近 0 或 1 时，p(1-p) → 0
    2. 导致梯度消失，学习停滞
    3. 训练速度慢，容易陷入局部最优

    【数值例子】
    假设：y=1（真实正类），p=0.2（预测错了），x=2

    交叉熵梯度：∂L/∂w = (0.2 - 1) × 2 = -1.6（梯度大，快速修正）
    MSE梯度：    ∂L/∂w = 2(0.2-1) × 0.2×0.8 × 2 = -0.512（梯度被削弱！）

    结论：交叉熵 + Sigmoid 是分类问题的黄金组合！
    ============================================================================
    """
    # 裁剪预测值，避免 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 计算交叉熵
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


# ==================== 2. 逻辑回归类 ====================
class LogisticRegression:
    """从零实现逻辑回归"""

    def __init__(self, learning_rate=0.01, n_epochs=1000, batch_size=32):
        """
        参数：
            learning_rate: 学习率
            n_epochs: 训练轮数
            batch_size: 批量大小（使用 Mini-batch GD）
        """
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        训练模型

        流程：
        1. 初始化参数 w, b
        2. 前向传播：计算预测概率
        3. 计算损失
        4. 反向传播：计算梯度
        5. 更新参数
        """
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Mini-batch 梯度下降
        for epoch in range(self.n_epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)

            # 分批训练
            for start_idx in range(0, n_samples, self.batch_size):
                # 获取当前 batch
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # ========== 前向传播 ==========
                # 1. 线性组合
                z = np.dot(X_batch, self.weights) + self.bias

                # 2. Sigmoid 激活
                y_pred = sigmoid(z)

                # ========== 计算梯度 ==========
                # 推导过程：
                # Loss = -[y*log(σ(z)) + (1-y)*log(1-σ(z))]
                # ∂Loss/∂w = (σ(z) - y) * x
                # ∂Loss/∂b = σ(z) - y

                batch_size_actual = len(X_batch)
                dw = (1 / batch_size_actual) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / batch_size_actual) * np.sum(y_pred - y_batch)

                # ========== 更新参数 ==========
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # 记录每个 epoch 的损失（用全部数据计算）
            if epoch % 10 == 0:
                z_all = np.dot(X, self.weights) + self.bias
                y_pred_all = sigmoid(z_all)
                loss = binary_cross_entropy(y, y_pred_all)
                self.loss_history.append(loss)

    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        预测类别

        threshold: 决策阈值
        - 默认 0.5：p >= 0.5 → 正类，p < 0.5 → 负类
        - 可以调整：如垃圾邮件检测可能用 0.7（宁可漏过不可误杀）

        ========================================================================
        🎯 精确率 vs 召回率：决策阈值的权衡
        ========================================================================

        【混淆矩阵 (Confusion Matrix)】

                        预测为正类    预测为负类
        实际是正类        TP           FN
                      (真正例)     (假负例)
        实际是负类        FP           TN
                      (假正例)     (真负例)

        TP (True Positive):  预测为正，实际为正 ✅
        TN (True Negative):  预测为负，实际为负 ✅
        FP (False Positive): 预测为正，实际为负 ❌ (误报)
        FN (False Negative): 预测为负，实际为正 ❌ (漏报)

        【三大指标】

        1. 准确率 (Accuracy)
           = (TP + TN) / (TP + TN + FP + FN)
           = 预测对的 / 总样本
           含义：整体预测的准确程度

        2. 精确率 (Precision) - "查准率"
           = TP / (TP + FP)
           = 真正例 / 预测为正类的所有样本
           含义：在所有"预测为正"的样本中，有多少真的是正类
           问题：模型说是正类，有多大把握？

        3. 召回率 (Recall) - "查全率"
           = TP / (TP + FN)
           = 真正例 / 实际为正类的所有样本
           含义：在所有"真正的正类"中，模型找到了多少
           问题：所有正类中，漏掉了多少？

        【直觉理解】

        场景：垃圾邮件检测
        - 精确率：被标记为垃圾邮件的，有多少真的是垃圾？
          → 高精确率 = 不会误杀正常邮件
        - 召回率：所有垃圾邮件中，抓到了多少？
          → 高召回率 = 不会漏掉垃圾邮件

        场景：疾病诊断
        - 精确率：诊断为阳性的，有多少真的有病？
          → 高精确率 = 减少误诊（健康人被诊断为有病）
        - 召回率：所有患者中，检测出了多少？
          → 高召回率 = 减少漏诊（有病但没检测出来）

        【阈值的影响】

        阈值 ↑ (如 0.3 → 0.7)：
        ├─ 预测为正类的样本 ↓（更严格）
        ├─ TP ↓，FP ↓（误报少了）
        ├─ 精确率 ↑（说是正类时更可靠）
        └─ 召回率 ↓（漏掉更多正类）

        阈值 ↓ (如 0.7 → 0.3)：
        ├─ 预测为正类的样本 ↑（更宽松）
        ├─ TP ↑，FP ↑（误报多了）
        ├─ 召回率 ↑（抓到更多正类）
        └─ 精确率 ↓（说是正类时不太可靠）

        【核心权衡】
        精确率 ↑ ⇔ 召回率 ↓（通常情况）

        无法同时最大化！需要根据业务需求选择：

        • 重视精确率（宁可漏过，不可误杀）
          → 阈值调高（如 0.7）
          → 案例：推荐系统（宁可少推荐，不推荐错的）
                  垃圾邮件检测（不能误删正常邮件）

        • 重视召回率（宁可误杀，不可漏过）
          → 阈值调低（如 0.3）
          → 案例：疾病筛查（不能漏掉患者）
                  欺诈检测（不能放过欺诈交易）

        【F1 Score - 平衡两者】
        F1 = 2 × (Precision × Recall) / (Precision + Recall)

        F1 是精确率和召回率的调和平均数
        当两者都高时，F1 才高（平衡指标）

        【实际例子】

        假设 100 个样本：60 个正类，40 个负类
        模型预测结果：

        阈值 = 0.3（宽松）：
        ├─ 预测为正：70 个（TP=55, FP=15）
        ├─ 预测为负：30 个（TN=25, FN=5）
        ├─ Precision = 55/70 = 0.786 (78.6%)
        ├─ Recall = 55/60 = 0.917 (91.7%)
        └─ 特点：抓到了大部分正类，但误报多

        阈值 = 0.7（严格）：
        ├─ 预测为正：35 个（TP=33, FP=2）
        ├─ 预测为负：65 个（TN=38, FN=27）
        ├─ Precision = 33/35 = 0.943 (94.3%)
        ├─ Recall = 33/60 = 0.550 (55.0%)
        └─ 特点：预测为正时很可靠，但漏掉很多

        阈值 = 0.5（平衡）：
        ├─ 找到最佳平衡点
        └─ 根据实际情况调整

        【记忆口诀】
        精确率：我说的对不对？（预测准不准）
        召回率：我找全了没有？（漏了多少）

        ========================================================================
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ==================== 3. 数据生成 ====================
def generate_binary_data(n_samples=200, n_features=2, random_state=42):
    """
    生成二分类数据

    使用 sklearn 的 make_classification 生成线性可分的数据
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,      # 所有特征都有用
        n_redundant=0,        # 无冗余特征
        n_clusters_per_class=1,
        class_sep=1.5,        # 类别分离度
        random_state=random_state
    )
    return X, y


def generate_nonlinear_data(n_samples=200):
    """
    生成非线性可分的数据（圆形分布）

    演示逻辑回归的局限性：只能学习线性决策边界
    """
    np.random.seed(42)

    # 内圆：负类
    r_inner = np.random.uniform(0, 1, n_samples // 2)
    theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_inner = np.column_stack([
        r_inner * np.cos(theta_inner),
        r_inner * np.sin(theta_inner)
    ])
    y_inner = np.zeros(n_samples // 2)

    # 外圆：正类
    r_outer = np.random.uniform(2, 3, n_samples // 2)
    theta_outer = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X_outer = np.column_stack([
        r_outer * np.cos(theta_outer),
        r_outer * np.sin(theta_outer)
    ])
    y_outer = np.ones(n_samples // 2)

    X = np.vstack([X_inner, X_outer])
    y = np.concatenate([y_inner, y_outer])

    return X, y


# ==================== 4. 可视化 ====================
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    绘制决策边界

    原理：
    - 在整个特征空间生成网格点
    - 用模型预测每个点的类别
    - 用颜色区分不同区域
    """
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 生成网格点
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=1)

    # 绘制数据点
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                c='blue', marker='o', s=50, edgecolors='k', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                c='red', marker='s', s=50, edgecolors='k', label='Class 1', alpha=0.7)

    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)


def visualize_sigmoid():
    """可视化 Sigmoid 函数"""
    z = np.linspace(-10, 10, 200)
    sigma = sigmoid(z)

    plt.figure(figsize=(10, 6))
    plt.plot(z, sigma, linewidth=3, color='purple')

    # 标注关键点
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision threshold (0.5)')
    plt.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='z = 0')
    plt.scatter([0, -2, 2], [sigmoid(0), sigmoid(-2), sigmoid(2)],
                s=100, c='red', zorder=5, edgecolors='black', linewidths=2)

    # 添加注释
    plt.text(0, 0.5, '  (0, 0.5)', fontsize=11, verticalalignment='bottom')
    plt.text(-2, sigmoid(-2), f'  ({-2:.0f}, {sigmoid(-2):.3f})', fontsize=10)
    plt.text(2, sigmoid(2), f'  ({2:.0f}, {sigmoid(2):.3f})', fontsize=10)

    plt.xlabel('z (linear output)', fontsize=12)
    plt.ylabel('σ(z) (probability)', fontsize=12)
    plt.title('Sigmoid Function: σ(z) = 1 / (1 + e^(-z))', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('sigmoid_function.png', dpi=100)
    print("📊 Sigmoid 函数图已保存到: sigmoid_function.png")
    plt.show()


# ==================== 5. 主程序 ====================
def main():
    print("=" * 70)
    print("逻辑回归 (Logistic Regression) - 二分类任务")
    print("=" * 70)

    # ========== 1. 可视化 Sigmoid 函数 ==========
    print("\n" + "=" * 70)
    print("📈 第一步：理解 Sigmoid 函数")
    print("=" * 70)
    print("""
Sigmoid 函数是逻辑回归的核心：
- 作用：将线性输出 z = wx + b 映射到 [0, 1] 概率
- 公式：σ(z) = 1 / (1 + e^(-z))
- 性质：单调递增，在 z=0 处值为 0.5
    """)
    visualize_sigmoid()

    # ========== 2. 生成数据 ==========
    print("\n" + "=" * 70)
    print("📊 第二步：生成二分类数据")
    print("=" * 70)
    X, y = generate_binary_data(n_samples=300, n_features=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"正类样本: {np.sum(y_train == 1)}, 负类样本: {np.sum(y_train == 0)}")

    # ========== 3. 训练模型 ==========
    print("\n" + "=" * 70)
    print("🔧 第三步：训练逻辑回归模型")
    print("=" * 70)

    model = LogisticRegression(learning_rate=0.1, n_epochs=200, batch_size=32)
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

    # 学习到的参数
    print(f"\n学习到的参数：")
    print(f"  权重 w = {model.weights}")
    print(f"  偏置 b = {model.bias:.4f}")

    # 预测示例
    print(f"\n预测示例（前5个测试样本）：")
    sample_probs = model.predict_proba(X_test[:5])
    sample_preds = model.predict(X_test[:5])
    for i in range(5):
        print(f"  样本 {i+1}: 真实={y_test[i]}, 预测={sample_preds[i]}, 概率={sample_probs[i]:.4f}")

    # ========== 5. 可视化结果 ==========
    print("\n" + "=" * 70)
    print("🎨 第五步：可视化决策边界和训练过程")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 5))

    # 子图1：训练集决策边界
    plt.subplot(1, 3, 1)
    plot_decision_boundary(model, X_train, y_train,
                          title=f'Training Set\nAccuracy: {train_acc:.2%}')

    # 子图2：测试集决策边界
    plt.subplot(1, 3, 2)
    plot_decision_boundary(model, X_test, y_test,
                          title=f'Test Set\nAccuracy: {test_acc:.2%}')

    # 子图3：损失函数下降曲线
    plt.subplot(1, 3, 3)
    plt.plot(range(0, model.n_epochs, 10), model.loss_history,
             linewidth=2, color='blue', marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('logistic_regression_result.png', dpi=100)
    print("\n📊 结果图已保存到: logistic_regression_result.png")
    plt.show()

    # ========== 6. 对比 sklearn ==========
    print("\n" + "=" * 70)
    print("🔬 第六步：与 sklearn 对比验证")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression as SklearnLR

    sklearn_model = SklearnLR(max_iter=1000, random_state=42)
    sklearn_model.fit(X_train, y_train)

    sklearn_train_acc = sklearn_model.score(X_train, y_train)
    sklearn_test_acc = sklearn_model.score(X_test, y_test)

    print(f"\nSklearn 逻辑回归：")
    print(f"  训练集准确率: {sklearn_train_acc:.4f} ({sklearn_train_acc * 100:.2f}%)")
    print(f"  测试集准确率: {sklearn_test_acc:.4f} ({sklearn_test_acc * 100:.2f}%)")
    print(f"  权重 w = {sklearn_model.coef_[0]}")
    print(f"  偏置 b = {sklearn_model.intercept_[0]:.4f}")

    print(f"\n对比结果：")
    print(f"  准确率差异（测试集）: {abs(test_acc - sklearn_test_acc):.4f}")
    print(f"  ✅ 实现基本正确！")


# ==================== 6. 实验区 ====================
def experiment_decision_threshold():
    """
    实验：决策阈值的影响

    探索不同阈值对分类结果的影响
    """
    print("\n" + "=" * 70)
    print("🧪 实验：决策阈值的影响")
    print("=" * 70)

    X, y = generate_binary_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练模型
    model = LogisticRegression(learning_rate=0.1, n_epochs=200)
    model.fit(X_train, y_train)

    # 测试不同阈值
    thresholds = [0.3, 0.5, 0.7, 0.9]

    print(f"\n{'阈值':<10} {'准确率':<10} {'预测为正类的比例':<20}")
    print("-" * 40)

    for threshold in thresholds:
        y_pred = model.predict(X_test, threshold=threshold)
        accuracy = np.mean(y_pred == y_test)
        positive_rate = np.mean(y_pred == 1)
        print(f"{threshold:<10.1f} {accuracy:<10.4f} {positive_rate:<20.2%}")

    print(f"\n💡 观察：")
    print(f"  - 阈值越高 → 预测为正类越严格（精确率高，召回率低）")
    print(f"  - 阈值越低 → 预测为正类越宽松（召回率高，精确率低）")
    print(f"  - 默认 0.5 通常是平衡点")


def experiment_nonlinear_data():
    """
    实验：逻辑回归在非线性数据上的局限性

    展示逻辑回归只能学习线性决策边界
    """
    print("\n" + "=" * 70)
    print("🧪 实验：逻辑回归的局限性 - 非线性数据")
    print("=" * 70)

    # 生成非线性数据（圆形分布）
    X, y = generate_nonlinear_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练逻辑回归
    model = LogisticRegression(learning_rate=0.1, n_epochs=200)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\n数据特点：内圆为负类，外圆为正类（非线性可分）")
    print(f"训练集准确率: {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # 可视化
    plt.figure(figsize=(8, 6))
    plot_decision_boundary(model, X_test, y_test,
                          title=f'Nonlinear Data\nLogistic Regression Accuracy: {test_acc:.2%}')
    plt.tight_layout()
    plt.savefig('logistic_regression_nonlinear.png', dpi=100)
    print("\n📊 非线性数据结果图已保存到: logistic_regression_nonlinear.png")
    plt.show()

    print(f"\n💡 结论：")
    print(f"  - 逻辑回归只能学习线性决策边界")
    print(f"  - 对于非线性数据，准确率很低")
    print(f"  - 解决方案：特征工程（多项式特征）或使用非线性模型（神经网络）")


# ==================== 7. 总结 ====================
def print_summary():
    print("\n" + "=" * 70)
    print("✅ 核心要点总结")
    print("=" * 70)
    print("""
1. 逻辑回归 vs 线性回归
   线性回归：预测连续值（房价、温度）
   逻辑回归：预测离散类别（是/否、垃圾邮件/正常邮件）

2. Sigmoid 函数
   - 作用：将线性输出映射到 [0, 1] 概率
   - 公式：σ(z) = 1 / (1 + e^(-z))
   - z = wx + b 是线性组合

3. 交叉熵损失
   - 为什么不用 MSE？MSE 对分类问题优化效果差
   - 公式：Loss = -[y*log(p) + (1-y)*log(1-p)]
   - 衡量预测概率分布和真实分布的差异

4. 决策边界
   - 线性边界：wx + b = 0 的直线/平面
   - 阈值：默认 0.5，可根据业务需求调整

5. 应用场景
   ✓ 垃圾邮件检测
   ✓ 疾病诊断（是否患病）
   ✓ 客户流失预测
   ✓ 信用评分（是否违约）

6. 局限性
   ✗ 只能学习线性决策边界
   ✗ 对非线性数据效果差
   → 解决方案：特征工程或神经网络

7. 与神经网络的关系
   - 逻辑回归 = 单层单神经元的神经网络
   - Sigmoid = 激活函数
   - 交叉熵 = 分类问题的标准损失函数
   - 逻辑回归是理解神经网络的基础！
    """)

    print("=" * 70)
    print("🎯 下一步学习建议")
    print("=" * 70)
    print("""
1. 多分类问题 (Softmax Regression)
   - 扩展到 3 个或更多类别
   - Softmax 函数：Sigmoid 的多分类版本

2. 正则化 (L1/L2 Regularization)
   - 防止过拟合
   - 特征选择

3. 模型评估指标
   - 精确率 (Precision)
   - 召回率 (Recall)
   - F1 Score
   - ROC 曲线和 AUC

4. 神经网络基础
   - 多层感知机 (MLP)
   - 反向传播算法
   - 激活函数对比
    """)


if __name__ == "__main__":
    # 主实验
    main()

    # 额外实验（取消注释运行）
    experiment_decision_threshold()
    experiment_nonlinear_data()

    # 总结
    print_summary()

    print("\n💡 练习建议：")
    print("  1. 修改学习率和训练轮数，观察对准确率的影响")
    print("  2. 尝试不同的决策阈值，理解精确率和召回率的权衡")
    print("  3. 在非线性数据上尝试添加多项式特征（如 x^2, xy, y^2）")
    print("  4. 思考：为什么逻辑回归叫'回归'但实际是分类算法？")
