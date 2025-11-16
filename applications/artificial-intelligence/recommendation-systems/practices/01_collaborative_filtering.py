"""
协同过滤 (Collaborative Filtering)

问题：如何基于用户的历史行为进行推荐？
目标：理解推荐系统的基础算法

核心概念：
1. 用户-物品矩阵：用户对物品的评分/交互
2. 相似度计算：找到相似的用户或物品
3. User-based CF：相似用户喜欢的，推荐给你
4. Item-based CF：你喜欢的物品相似的，推荐给你
5. 矩阵分解：学习用户和物品的隐向量

为什么重要？
- 协同过滤是推荐系统的基础
- 理解相似度计算（后面双塔模型也会用）
- 矩阵分解是 Embedding 的雏形
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# ==================== 1. 用户-物品评分矩阵 ====================
def create_rating_matrix():
    """
    创建用户-物品评分矩阵

    ====================================================================
    🔑 什么是用户-物品矩阵？
    ====================================================================

    用户-物品矩阵 = 记录用户对物品的评分/交互

    例子：电影评分（1-5分）

                电影1  电影2  电影3  电影4  电影5
    用户A         5      3      ?      1      ?
    用户B         4      ?      ?      3      5
    用户C         1      1      ?      5      4
    用户D         ?      ?      5      4      ?
    用户E         ?      4      5      ?      2

    特点：
    - 行 = 用户
    - 列 = 物品
    - 值 = 评分（? 表示未评分，稀疏矩阵）

    推荐系统的目标：
    预测 ? 的位置（用户可能喜欢什么）

    ====================================================================
    🔑 真实世界的评分矩阵
    ====================================================================

    规模：
    - 淘宝：数亿用户 × 数亿商品
    - Netflix：1亿用户 × 数万电影
    - 抖音：数亿用户 × 数千万视频

    稀疏性：
    - 99.9%的位置都是空的（用户不可能看过所有商品）
    - 这就是"稀疏矩阵"问题

    隐式反馈 vs 显式反馈：
    - 显式：用户明确评分（1-5星）
    - 隐式：点击、浏览、购买（没有明确分数）

    ====================================================================
    """

    # 示例评分矩阵（5用户 × 6电影，1-5分，0表示未评分）
    ratings = np.array([
        [5, 3, 0, 1, 0, 2],  # 用户0
        [4, 0, 0, 3, 5, 0],  # 用户1
        [1, 1, 0, 5, 4, 0],  # 用户2
        [0, 0, 5, 4, 0, 3],  # 用户3
        [0, 4, 5, 0, 2, 4],  # 用户4
    ], dtype=float)

    return ratings


def visualize_rating_matrix(ratings):
    """可视化评分矩阵"""
    print("=" * 70)
    print("用户-物品评分矩阵")
    print("=" * 70)

    n_users, n_items = ratings.shape

    print(f"\n矩阵大小: {n_users} 用户 × {n_items} 物品")
    print(f"\n评分矩阵:")
    print(f"{'用户':<8}", end='')
    for j in range(n_items):
        print(f"电影{j:<6}", end='')
    print()

    for i in range(n_users):
        print(f"用户{i:<6}", end='')
        for j in range(n_items):
            if ratings[i, j] == 0:
                print(f"{'?':<8}", end='')
            else:
                print(f"{ratings[i, j]:<8.0f}", end='')
        print()

    # 统计
    total_entries = n_users * n_items
    rated_entries = np.count_nonzero(ratings)
    sparsity = (1 - rated_entries / total_entries) * 100

    print(f"\n统计信息:")
    print(f"  总共: {total_entries} 个位置")
    print(f"  已评分: {rated_entries} 个")
    print(f"  未评分: {total_entries - rated_entries} 个")
    print(f"  稀疏度: {sparsity:.1f}% （越大越稀疏）")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建掩码（0的位置用灰色表示）
    masked_ratings = np.ma.masked_where(ratings == 0, ratings)

    im = ax.imshow(masked_ratings, cmap='YlOrRd', vmin=1, vmax=5)

    # 设置刻度
    ax.set_xticks(range(n_items))
    ax.set_yticks(range(n_users))
    ax.set_xticklabels([f'电影{i}' for i in range(n_items)])
    ax.set_yticklabels([f'用户{i}' for i in range(n_users)])

    # 添加数值标签
    for i in range(n_users):
        for j in range(n_items):
            if ratings[i, j] > 0:
                text = ax.text(j, i, f'{ratings[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=12)
            else:
                text = ax.text(j, i, '?',
                             ha="center", va="center", color="gray", fontsize=12)

    ax.set_title('用户-物品评分矩阵\n(? = 未评分，推荐系统要预测这些位置)',
                fontsize=12, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('评分 (1-5)', fontsize=10)

    plt.tight_layout()
    plt.savefig('rating_matrix.png', dpi=100, bbox_inches='tight')
    print("\n📊 评分矩阵可视化已保存: rating_matrix.png")
    plt.close()


# ==================== 2. User-based 协同过滤 ====================
class UserBasedCF:
    """
    基于用户的协同过滤

    ====================================================================
    🔑 User-based CF 的核心思想
    ====================================================================

    "物以类聚，人以群分"

    核心逻辑：
    1. 找到和你相似的用户（有相似的评分模式）
    2. 他们喜欢的东西，推荐给你

    例子：
    - 你和用户B都喜欢动作片、科幻片
    - 用户B最近看了《星际穿越》，评分5分
    - 你还没看过《星际穿越》
    - 系统推荐给你！

    ====================================================================
    🔑 如何计算用户相似度？
    ====================================================================

    常用方法：余弦相似度

    用户A的评分向量: [5, 3, 0, 1, 0, 2]
    用户B的评分向量: [4, 0, 0, 3, 5, 0]

    相似度 = cos(θ) = (A · B) / (||A|| × ||B||)

    只考虑两人都评过分的物品：
    用户A: [5, 1]  (电影0和电影3)
    用户B: [4, 3]

    相似度 = (5×4 + 1×3) / (√(25+1) × √(16+9))
           = 23 / (5.1 × 5.0)
           = 0.90  (很相似！)

    ====================================================================
    🔑 如何预测评分？
    ====================================================================

    加权平均：相似用户的评分加权平均

    预测用户A对电影4的评分：

    找到相似用户：
    - 用户B (相似度0.9) 对电影4评分5
    - 用户C (相似度0.3) 对电影4评分4

    预测 = (0.9×5 + 0.3×4) / (0.9 + 0.3)
         = (4.5 + 1.2) / 1.2
         = 4.75

    ====================================================================
    """

    def __init__(self, ratings):
        """
        Parameters:
            ratings: 用户-物品评分矩阵 (n_users, n_items)
        """
        self.ratings = ratings.copy()
        self.n_users, self.n_items = ratings.shape

        # 计算用户相似度矩阵
        self.user_similarity = self._compute_similarity()

    def _compute_similarity(self):
        """
        计算用户之间的相似度

        使用余弦相似度，只考虑两个用户都评过分的物品
        """
        print("\n计算用户相似度...")

        # 创建相似度矩阵
        similarity = np.zeros((self.n_users, self.n_users))

        for i in range(self.n_users):
            for j in range(i, self.n_users):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    # 找到两个用户都评过分的物品
                    mask = (self.ratings[i] > 0) & (self.ratings[j] > 0)

                    if np.sum(mask) > 0:  # 有共同评分
                        # 提取共同评分
                        vec_i = self.ratings[i][mask]
                        vec_j = self.ratings[j][mask]

                        # 余弦相似度
                        dot_product = np.dot(vec_i, vec_j)
                        norm_i = np.linalg.norm(vec_i)
                        norm_j = np.linalg.norm(vec_j)

                        if norm_i > 0 and norm_j > 0:
                            similarity[i, j] = dot_product / (norm_i * norm_j)
                            similarity[j, i] = similarity[i, j]  # 对称

        print(f"用户相似度矩阵计算完成: {similarity.shape}")
        return similarity

    def predict(self, user_id, item_id, k=3):
        """
        预测用户对物品的评分

        Parameters:
            user_id: 用户ID
            item_id: 物品ID
            k: 使用最相似的k个用户

        Returns:
            预测评分
        """
        # 如果用户已经评过分，返回原评分
        if self.ratings[user_id, item_id] > 0:
            return self.ratings[user_id, item_id]

        # 找到对该物品评过分的用户
        rated_users = np.where(self.ratings[:, item_id] > 0)[0]

        if len(rated_users) == 0:
            # 没人评过分，返回该用户的平均分
            user_ratings = self.ratings[user_id][self.ratings[user_id] > 0]
            return user_ratings.mean() if len(user_ratings) > 0 else 3.0

        # 获取这些用户与目标用户的相似度
        similarities = self.user_similarity[user_id, rated_users]

        # 选择top-k相似用户
        top_k_idx = np.argsort(similarities)[-k:]
        top_k_users = rated_users[top_k_idx]
        top_k_sims = similarities[top_k_idx]

        # 如果相似度都是0，返回平均分
        if np.sum(top_k_sims) == 0:
            user_ratings = self.ratings[user_id][self.ratings[user_id] > 0]
            return user_ratings.mean() if len(user_ratings) > 0 else 3.0

        # 加权平均
        weighted_sum = np.sum(top_k_sims * self.ratings[top_k_users, item_id])
        prediction = weighted_sum / np.sum(top_k_sims)

        return prediction

    def recommend(self, user_id, n_recommendations=3):
        """
        为用户推荐物品

        Parameters:
            user_id: 用户ID
            n_recommendations: 推荐数量

        Returns:
            推荐的物品ID列表
        """
        # 找到用户未评分的物品
        unrated_items = np.where(self.ratings[user_id] == 0)[0]

        if len(unrated_items) == 0:
            return []

        # 预测所有未评分物品的分数
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))

        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)

        # 返回top-n
        return [item_id for item_id, _ in predictions[:n_recommendations]]


# ==================== 3. Item-based 协同过滤 ====================
class ItemBasedCF:
    """
    基于物品的协同过滤

    ====================================================================
    🔑 Item-based CF 的核心思想
    ====================================================================

    "你喜欢这个，那也会喜欢那个"

    核心逻辑：
    1. 找到和你喜欢的物品相似的其他物品
    2. 推荐给你

    例子：
    - 你喜欢《泰坦尼克号》（爱情片）
    - 《罗马假日》和《泰坦尼克号》很相似（都是爱情片）
    - 系统推荐《罗马假日》给你

    ====================================================================
    🔑 User-based vs Item-based
    ====================================================================

    User-based:
    - 找相似用户
    - 适合用户少、物品多的场景
    - 用户兴趣变化快，需要经常重新计算

    Item-based:
    - 找相似物品
    - 适合物品少、用户多的场景（大多数场景！）
    - 物品相似度相对稳定，可以离线计算
    - ✅ 工业界更常用！

    为什么Item-based更常用？
    1. 淘宝：10亿用户 vs 10亿商品
       → 用户太多，计算用户相似度太慢
       → 商品相似度可以离线计算，查询快

    2. 可解释性：
       "因为你买了iPhone，推荐AirPods" ✅ 用户容易理解
       "因为用户123456和你相似，推荐..." ❌ 用户不理解

    ====================================================================
    🔑 如何计算物品相似度？
    ====================================================================

    和User-based类似，但是换个角度：

    物品A的评分向量: [5, 4, 1, 0, 0]  (5个用户对它的评分)
    物品B的评分向量: [3, 0, 1, 0, 4]

    相似度 = 只考虑都评过分的用户

    ====================================================================
    """

    def __init__(self, ratings):
        self.ratings = ratings.copy()
        self.n_users, self.n_items = ratings.shape

        # 计算物品相似度矩阵
        self.item_similarity = self._compute_similarity()

    def _compute_similarity(self):
        """计算物品之间的相似度"""
        print("\n计算物品相似度...")

        similarity = np.zeros((self.n_items, self.n_items))

        for i in range(self.n_items):
            for j in range(i, self.n_items):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    # 找到对两个物品都评过分的用户
                    mask = (self.ratings[:, i] > 0) & (self.ratings[:, j] > 0)

                    if np.sum(mask) > 0:
                        vec_i = self.ratings[:, i][mask]
                        vec_j = self.ratings[:, j][mask]

                        dot_product = np.dot(vec_i, vec_j)
                        norm_i = np.linalg.norm(vec_i)
                        norm_j = np.linalg.norm(vec_j)

                        if norm_i > 0 and norm_j > 0:
                            similarity[i, j] = dot_product / (norm_i * norm_j)
                            similarity[j, i] = similarity[i, j]

        print(f"物品相似度矩阵计算完成: {similarity.shape}")
        return similarity

    def predict(self, user_id, item_id, k=3):
        """预测评分（逻辑和User-based类似，但使用物品相似度）"""
        if self.ratings[user_id, item_id] > 0:
            return self.ratings[user_id, item_id]

        # 找到用户评过分的物品
        rated_items = np.where(self.ratings[user_id] > 0)[0]

        if len(rated_items) == 0:
            return 3.0

        # 获取这些物品与目标物品的相似度
        similarities = self.item_similarity[item_id, rated_items]

        # Top-k相似物品
        top_k_idx = np.argsort(similarities)[-k:]
        top_k_items = rated_items[top_k_idx]
        top_k_sims = similarities[top_k_idx]

        if np.sum(top_k_sims) == 0:
            return 3.0

        # 加权平均
        weighted_sum = np.sum(top_k_sims * self.ratings[user_id, top_k_items])
        prediction = weighted_sum / np.sum(top_k_sims)

        return prediction

    def recommend(self, user_id, n_recommendations=3):
        """推荐物品"""
        unrated_items = np.where(self.ratings[user_id] == 0)[0]

        if len(unrated_items) == 0:
            return []

        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return [item_id for item_id, _ in predictions[:n_recommendations]]


# ==================== 4. 矩阵分解 (Matrix Factorization) ====================
class MatrixFactorization:
    """
    矩阵分解 (Matrix Factorization)

    ====================================================================
    🔑 什么是矩阵分解？
    ====================================================================

    核心思想：把大矩阵分解成两个小矩阵的乘积

    评分矩阵 R (m×n) ≈ U (m×k) × I (k×n)
                       ↑           ↑
                   用户矩阵    物品矩阵

    例子：
    5个用户 × 6个物品，分解成 k=2 个隐因子

    R (5×6) ≈ U (5×2) × I (2×6)

    用户矩阵 U:
    [0.9, 0.1]  ← 用户0喜欢因子0（动作片）
    [0.8, 0.2]  ← 用户1也喜欢动作片
    [0.1, 0.9]  ← 用户2喜欢因子1（爱情片）
    ...

    物品矩阵 I:
    [0.9, 0.8, 0.1, ...]  ← 因子0的物品分布
    [0.1, 0.2, 0.9, ...]  ← 因子1的物品分布

    预测评分：
    用户0对物品2的评分 = U[0] · I[:, 2]
                       = [0.9, 0.1] · [0.1, 0.9]
                       = 0.9×0.1 + 0.1×0.9
                       = 0.18

    ====================================================================
    🔑 为什么矩阵分解有效？
    ====================================================================

    1. 降维：
       原始：5×6 = 30个数（很多是空的）
       分解：5×2 + 2×6 = 22个数（更紧凑）

    2. 学习隐因子：
       自动学习用户和物品的隐藏特征
       - 因子0可能代表"动作片"
       - 因子1可能代表"爱情片"
       - ...

    3. 泛化能力强：
       可以预测没见过的用户-物品组合

    4. 这就是 Embedding 的前身！
       U 和 I 就是用户和物品的 Embedding

    ====================================================================
    🔑 如何训练？
    ====================================================================

    优化目标：最小化预测误差

    Loss = Σ (R_ij - U_i · I_j)²  (对所有已评分的位置)

    梯度下降：
    ∂Loss/∂U = -2 × (R_ij - U_i · I_j) × I_j
    ∂Loss/∂I = -2 × (R_ij - U_i · I_j) × U_i

    更新：
    U_i = U_i - lr × ∂Loss/∂U
    I_j = I_j - lr × ∂Loss/∂I

    ====================================================================
    """

    def __init__(self, ratings, n_factors=5, learning_rate=0.01, n_epochs=100, reg=0.01):
        """
        Parameters:
            ratings: 评分矩阵
            n_factors: 隐因子数量（Embedding维度）
            learning_rate: 学习率
            n_epochs: 训练轮数
            reg: 正则化系数（防止过拟合）
        """
        self.ratings = ratings.copy()
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.reg = reg

        # 初始化用户和物品矩阵（随机小值）
        self.user_factors = np.random.randn(self.n_users, n_factors) * 0.1
        self.item_factors = np.random.randn(self.n_items, n_factors) * 0.1

        self.loss_history = []

    def train(self):
        """训练矩阵分解模型"""
        print(f"\n训练矩阵分解 ({self.n_factors}个隐因子)...")

        # 找到所有已评分的位置
        rated_mask = self.ratings > 0
        rated_indices = np.where(rated_mask)

        for epoch in range(self.n_epochs):
            total_loss = 0

            # 遍历所有已评分的位置
            for idx in range(len(rated_indices[0])):
                i = rated_indices[0][idx]  # 用户
                j = rated_indices[1][idx]  # 物品

                # 预测评分
                pred = np.dot(self.user_factors[i], self.item_factors[j])

                # 误差
                error = self.ratings[i, j] - pred
                total_loss += error ** 2

                # 梯度下降（带L2正则化）
                user_grad = -2 * error * self.item_factors[j] + 2 * self.reg * self.user_factors[i]
                item_grad = -2 * error * self.user_factors[i] + 2 * self.reg * self.item_factors[j]

                self.user_factors[i] -= self.lr * user_grad
                self.item_factors[j] -= self.lr * item_grad

            # 记录损失
            avg_loss = total_loss / len(rated_indices[0])
            self.loss_history.append(avg_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        print("训练完成！")

    def predict(self, user_id, item_id):
        """预测评分"""
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])

    def recommend(self, user_id, n_recommendations=3):
        """推荐物品"""
        unrated_items = np.where(self.ratings[user_id] == 0)[0]

        if len(unrated_items) == 0:
            return []

        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return [item_id for item_id, _ in predictions[:n_recommendations]]


# ==================== 5. 对比三种方法 ====================
def compare_methods():
    """对比三种协同过滤方法"""
    print("\n" + "=" * 70)
    print("对比实验: User-based vs Item-based vs Matrix Factorization")
    print("=" * 70)

    # 创建评分矩阵
    ratings = create_rating_matrix()
    visualize_rating_matrix(ratings)

    # 1. User-based CF
    print("\n" + "=" * 70)
    print("1. User-based 协同过滤")
    print("=" * 70)
    user_cf = UserBasedCF(ratings)

    # 可视化用户相似度
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(user_cf.user_similarity, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0].set_title('用户相似度矩阵\n(User-based CF)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('用户')
    axes[0].set_ylabel('用户')
    plt.colorbar(im1, ax=axes[0])

    # 为用户0推荐
    user_id = 0
    recommendations = user_cf.recommend(user_id, n_recommendations=3)
    print(f"\n为用户{user_id}推荐物品: {recommendations}")

    # 预测一些评分
    print(f"\n预测评分:")
    for item_id in [2, 4, 5]:
        if ratings[user_id, item_id] == 0:
            pred = user_cf.predict(user_id, item_id)
            print(f"  用户{user_id} 对 电影{item_id} 的预测评分: {pred:.2f}")

    # 2. Item-based CF
    print("\n" + "=" * 70)
    print("2. Item-based 协同过滤")
    print("=" * 70)
    item_cf = ItemBasedCF(ratings)

    im2 = axes[1].imshow(item_cf.item_similarity, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title('物品相似度矩阵\n(Item-based CF)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('物品')
    axes[1].set_ylabel('物品')
    plt.colorbar(im2, ax=axes[1])

    recommendations = item_cf.recommend(user_id, n_recommendations=3)
    print(f"\n为用户{user_id}推荐物品: {recommendations}")

    print(f"\n预测评分:")
    for item_id in [2, 4, 5]:
        if ratings[user_id, item_id] == 0:
            pred = item_cf.predict(user_id, item_id)
            print(f"  用户{user_id} 对 电影{item_id} 的预测评分: {pred:.2f}")

    # 3. Matrix Factorization
    print("\n" + "=" * 70)
    print("3. 矩阵分解 (Matrix Factorization)")
    print("=" * 70)
    mf = MatrixFactorization(ratings, n_factors=3, learning_rate=0.05, n_epochs=200, reg=0.01)
    mf.train()

    # 可视化损失曲线
    axes[2].plot(mf.loss_history, linewidth=2, color='#3498db')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].set_title('训练损失曲线\n(Matrix Factorization)', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('cf_comparison.png', dpi=100, bbox_inches='tight')
    print("\n📊 对比结果已保存: cf_comparison.png")
    plt.close()

    recommendations = mf.recommend(user_id, n_recommendations=3)
    print(f"\n为用户{user_id}推荐物品: {recommendations}")

    print(f"\n预测评分:")
    for item_id in [2, 4, 5]:
        if ratings[user_id, item_id] == 0:
            pred = mf.predict(user_id, item_id)
            print(f"  用户{user_id} 对 电影{item_id} 的预测评分: {pred:.2f}")

    # 可视化学习到的用户和物品因子
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 用户因子
    im1 = axes[0].imshow(mf.user_factors, cmap='RdBu', aspect='auto')
    axes[0].set_title('用户隐因子矩阵 (User Embeddings)\n每行是一个用户的向量',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('隐因子维度')
    axes[0].set_ylabel('用户ID')
    axes[0].set_yticks(range(mf.n_users))
    plt.colorbar(im1, ax=axes[0])

    # 物品因子
    im2 = axes[1].imshow(mf.item_factors.T, cmap='RdBu', aspect='auto')
    axes[1].set_title('物品隐因子矩阵 (Item Embeddings)\n每列是一个物品的向量',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('物品ID')
    axes[1].set_ylabel('隐因子维度')
    axes[1].set_xticks(range(mf.n_items))
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('mf_embeddings.png', dpi=100, bbox_inches='tight')
    print("📊 Embedding可视化已保存: mf_embeddings.png")
    plt.close()


# ==================== 6. 主程序 ====================
def main():
    print("=" * 70)
    print("协同过滤 (Collaborative Filtering)")
    print("=" * 70)

    # 对比三种方法
    compare_methods()

    # 总结
    print("\n" + "=" * 70)
    print("✅ 核心知识点总结")
    print("=" * 70)
    print("""
1. 协同过滤的核心思想
   - 利用群体智慧进行推荐
   - 不需要物品的内容信息
   - 基于"相似性"进行预测

2. User-based CF（基于用户）

   步骤：
   a) 计算用户相似度（余弦相似度）
   b) 找到最相似的k个用户
   c) 用他们的评分加权平均

   公式：
   pred(u, i) = Σ sim(u, v) × r(v, i) / Σ sim(u, v)

   优点：
   ✓ 直观易理解
   ✓ 可以发现新兴趣

   缺点：
   ✗ 用户多时计算量大
   ✗ 用户兴趣变化快，需要频繁更新

3. Item-based CF（基于物品）⭐ 工业界常用

   步骤：
   a) 计算物品相似度
   b) 找到用户喜欢的物品的相似物品
   c) 推荐相似物品

   优点：
   ✓ 物品相似度相对稳定，可离线计算
   ✓ 扩展性好（用户多时效率高）
   ✓ 可解释性强："因为你喜欢A，推荐B"

   缺点：
   ✗ 推荐结果相似度高（容易陷入"信息茧房"）
   ✗ 对新物品不友好（冷启动）

4. Matrix Factorization（矩阵分解）⭐ 最重要！

   核心：R ≈ U × I^T

   - R: 评分矩阵 (m×n)
   - U: 用户因子 (m×k) ← 用户Embedding!
   - I: 物品因子 (n×k) ← 物品Embedding!
   - k: 隐因子数量（维度）

   预测：
   r_ui = u_u · i_i  (点积)

   训练：
   Loss = Σ (r_ui - u_u · i_i)² + λ(||u_u||² + ||i_i||²)
                ↑                        ↑
            预测误差                  L2正则化

   优点：
   ✓ 自动学习隐因子（不需要手动定义特征）
   ✓ 泛化能力强
   ✓ 可以处理稀疏矩阵
   ✓ 这就是Embedding的前身！

   这个就是深度学习推荐系统的基础：
   - U 和 I 就是 Embedding
   - 点积就是相似度计算
   - 这就是后面"双塔模型"的核心思想！

5. 相似度计算方法

   余弦相似度：最常用
   sim(A, B) = (A · B) / (||A|| × ||B||)
   范围：[-1, 1]，1=完全相同，-1=完全相反

   皮尔逊相关系数：考虑评分偏差
   sim(A, B) = Σ(r_A - r̄_A)(r_B - r̄_B) / √[Σ(r_A - r̄_A)² × Σ(r_B - r̄_B)²]

   欧氏距离：
   dist(A, B) = ||A - B||
   注意：距离越小越相似（和相似度相反）

6. 协同过滤的问题

   ❌ 冷启动（Cold Start）：
   - 新用户：没有历史行为，无法推荐
   - 新物品：没人评过分，无法被推荐

   解决：
   - 基于内容的推荐（Content-based）
   - 混合推荐（Hybrid）
   - 利用用户画像/物品特征

   ❌ 数据稀疏（Sparsity）：
   - 99%的位置都是空的
   - 很难找到足够的共同评分

   解决：
   - 矩阵分解（降维）
   - 深度学习方法

   ❌ 可扩展性（Scalability）：
   - 亿级用户×物品，计算量巨大

   解决：
   - 近似最近邻（ANN）
   - 聚类
   - 采样

7. 从协同过滤到深度学习推荐

   演进路径：

   协同过滤 → 矩阵分解 → 深度学习

   矩阵分解 (2006):
   R ≈ U × I^T

   → 加深度学习 (2016+):
   R ≈ f(U, I)  ← f是神经网络

   → 双塔模型:
   User Tower → user_embedding
   Item Tower → item_embedding
   score = user_embedding · item_embedding

   → 更复杂的模型:
   DIN, DIEN, DeepFM, Wide&Deep...

   核心思想一脉相承：
   - 学习用户和物品的Embedding
   - 计算相似度/得分
   - 优化预测准确性

8. 工业界实践

   Netflix Prize (2006-2009):
   - 矩阵分解大放异彩
   - Simon Funk's SVD算法

   Amazon:
   - Item-based CF为主
   - "购买此商品的用户还购买了..."

   YouTube:
   - 深度学习推荐系统
   - 双塔结构（召回）+ 精排

   淘宝:
   - 多路召回（协同过滤、内容、热门...）
   - 深度学习精排
   - 实时个性化

9. 实践建议

   如果你要做推荐系统：

   第一步：协同过滤 baseline
   - 快速验证想法
   - 理解数据特点

   第二步：矩阵分解
   - 引入隐因子
   - 提升效果

   第三步：深度学习
   - 加入更多特征
   - 双塔模型
   - 复杂交互

   记住：
   - 简单模型往往更robust
   - 不要过早优化
   - 数据质量 > 模型复杂度

10. 下一步学习：双塔模型

    协同过滤帮你理解了：
    ✓ 相似度计算
    ✓ Embedding的概念（矩阵分解）
    ✓ 用户-物品交互

    下一步：双塔模型
    - 用神经网络学习Embedding
    - 训练和推理分离
    - 工业界标准架构

    你已经做好准备了！🚀
    """)


if __name__ == "__main__":
    main()

    print("\n💡 练习建议:")
    print("  1. 尝试不同的相似度计算方法")
    print("  2. 调整矩阵分解的隐因子数量，观察效果")
    print("  3. 在真实数据集上测试（MovieLens）")
    print("  4. 思考：如何解决冷启动问题？")
    print("  5. 理解矩阵分解和Embedding的关系")
