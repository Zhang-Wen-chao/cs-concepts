"""
重排（Re-Ranking）- PyTorch实现
优化推荐列表的整体质量：多样性、打散、用户体验

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 重排在推荐链路中的位置
====================================================================

完整链路：
召回（几千） → 粗排（几百） → 精排（几十） → 重排 → 混排 → 展示

精排的问题：
- 只考虑单个物品的点击率（逐个打分）
- 忽略了列表级的目标（整体用户体验）
- 可能导致：
  ❌ 推荐结果同质化（都是同类商品）
  ❌ 同一作者/店铺重复出现
  ❌ 用户感觉"千篇一律"

重排的目标：
✓ 考虑列表级目标：多样性、新颖性
✓ 打散规则：类别打散、作者打散、价格打散
✓ 优化整体用户体验

====================================================================
🎯 重排的核心思想
====================================================================

从"单点优化"到"列表优化"：

精排：
- 输入：单个物品
- 输出：点击概率
- 目标：max P(click | user, item)

重排：
- 输入：物品列表
- 输出：重新排序的列表
- 目标：max 整体价值（多样性 + 相关性）

常用方法：

1️⃣ MMR (Maximal Marginal Relevance)
   - 最大边际相关性
   - 平衡相关性和多样性
   - 贪心选择：每次选最能增加多样性的物品

2️⃣ DPP (Determinantal Point Process)
   - 行列式点过程
   - 基于概率模型的多样性
   - 考虑物品之间的相似度矩阵

3️⃣ 规则打散
   - 类别打散：相邻物品不同类
   - 作者/店铺打散：避免连续推荐同一作者
   - 价格打散：高低价交替

====================================================================
🏗️ 本实现：三种重排方法
====================================================================

方法1：MMR（最大边际相关性）
- score = λ × relevance - (1-λ) × max_similarity
- 贪心选择：每次选相关性高且与已选物品最不相似的

方法2：规则打散
- 类别打散：相邻物品不同类
- 价格打散：高低价交替
- 简单有效

方法3：DPP（行列式点过程）
- 构造相似度矩阵 K
- 选择子集 Y 使得 det(K_Y) 最大
- 理论最优，但计算复杂

====================================================================
📊 评价指标
====================================================================

1. 多样性指标
   - ILD (Intra-List Diversity)：列表内物品的平均差异度
   - Coverage：覆盖的类别数

2. 相关性指标
   - NDCG：考虑位置的排序质量
   - Precision@K：前K个的准确率

3. 用户体验指标
   - 用户满意度
   - 停留时长
   - 跳出率

目标：
- 多样性提升 > 20%
- 相关性损失 < 5%

====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# ============ 1. 模拟精排输出 ============

class RankedItem:
    """精排后的物品"""

    def __init__(self, item_id, score, category, price, embedding):
        self.item_id = item_id
        self.score = score  # 精排分数（点击概率）
        self.category = category
        self.price = price
        self.embedding = embedding  # 物品向量（用于计算相似度）


def generate_ranked_items(num_items=50, num_categories=10, embedding_dim=16):
    """生成精排后的物品列表"""
    items = []
    for i in range(num_items):
        item_id = i
        score = np.random.uniform(0.3, 0.9)  # 精排分数
        category = np.random.randint(0, num_categories)
        price = np.random.uniform(10, 1000)
        embedding = np.random.randn(embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # 归一化

        items.append(RankedItem(item_id, score, category, price, embedding))

    # 按精排分数排序
    items.sort(key=lambda x: x.score, reverse=True)
    return items


# ============ 2. 重排方法 ============

class MMRReranker:
    """方法1：MMR (Maximal Marginal Relevance)

    MMR 公式：
    score(item) = λ × relevance(item) - (1-λ) × max_similarity(item, selected)

    λ: 相关性权重
    - λ=1: 只看相关性（等同于精排）
    - λ=0: 只看多样性
    - λ=0.5: 平衡
    """

    def __init__(self, lambda_param=0.5):
        self.lambda_param = lambda_param

    def rerank(self, items, top_k=20):
        """
        参数:
            items: 精排后的物品列表（已按分数排序）
            top_k: 重排后返回的物品数

        返回:
            reranked_items: 重排后的物品列表
        """
        selected = []
        remaining = items.copy()

        for _ in range(min(top_k, len(items))):
            if len(selected) == 0:
                # 第一个：选精排分数最高的
                best_item = remaining[0]
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                # 计算每个候选物品的 MMR 分数
                mmr_scores = []
                for item in remaining:
                    # 相关性：精排分数
                    relevance = item.score

                    # 多样性：与已选物品的最大相似度
                    similarities = [
                        cosine_similarity(
                            item.embedding.reshape(1, -1),
                            selected_item.embedding.reshape(1, -1)
                        )[0][0]
                        for selected_item in selected
                    ]
                    max_similarity = max(similarities)

                    # MMR 分数
                    mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                    mmr_scores.append((item, mmr_score))

                # 选择 MMR 分数最高的
                best_item, _ = max(mmr_scores, key=lambda x: x[1])
                selected.append(best_item)
                remaining.remove(best_item)

        return selected


class RuleBasedReranker:
    """方法2：规则打散

    规则：
    1. 类别打散：相邻物品不同类
    2. 价格打散：高低价交替
    """

    def __init__(self):
        pass

    def rerank(self, items, top_k=20):
        """
        贪心策略：
        - 每次选择分数最高且满足打散规则的物品
        """
        selected = []
        remaining = items.copy()

        for _ in range(min(top_k, len(items))):
            if len(selected) == 0:
                # 第一个：选精排分数最高的
                best_item = remaining[0]
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                # 找到分数最高且满足打散规则的物品
                last_item = selected[-1]

                # 候选：类别不同的物品
                candidates = [
                    item for item in remaining
                    if item.category != last_item.category
                ]

                if len(candidates) == 0:
                    # 如果没有不同类别的，放宽限制
                    candidates = remaining

                # 从候选中选分数最高的
                best_item = max(candidates, key=lambda x: x.score)
                selected.append(best_item)
                remaining.remove(best_item)

        return selected


class DPPReranker:
    """方法3：DPP (Determinantal Point Process)

    DPP 核心思想：
    - 构造核矩阵 K：K[i,j] = quality[i] × quality[j] × similarity[i,j]
    - 选择子集 Y 使得 det(K_Y) 最大
    - det 越大 → 多样性越好

    简化实现：贪心近似算法
    """

    def __init__(self):
        pass

    def rerank(self, items, top_k=20):
        """
        贪心 DPP：
        - 每次选择能最大化 det(K_Y) 的物品
        """
        selected = []
        remaining = items.copy()

        # 构造质量向量
        qualities = np.array([item.score for item in items])

        # 构造相似度矩阵
        embeddings = np.array([item.embedding for item in items])
        similarity_matrix = cosine_similarity(embeddings)

        for _ in range(min(top_k, len(items))):
            if len(selected) == 0:
                # 第一个：选精排分数最高的
                best_item = remaining[0]
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                # 计算每个候选物品的边际增益
                best_gain = -float('inf')
                best_item = None

                selected_indices = [items.index(item) for item in selected]

                for item in remaining:
                    item_idx = items.index(item)
                    candidate_indices = selected_indices + [item_idx]

                    # 构造子集的核矩阵
                    K_sub = np.outer(qualities[candidate_indices], qualities[candidate_indices])
                    K_sub *= similarity_matrix[np.ix_(candidate_indices, candidate_indices)]

                    # 计算行列式（多样性）
                    det = np.linalg.det(K_sub)

                    if det > best_gain:
                        best_gain = det
                        best_item = item

                selected.append(best_item)
                remaining.remove(best_item)

        return selected


# ============ 3. 评价指标 ============

def calculate_diversity(items):
    """计算列表的多样性（ILD - Intra-List Diversity）

    ILD = 平均物品间的差异度
    """
    if len(items) <= 1:
        return 0.0

    embeddings = np.array([item.embedding for item in items])
    similarity_matrix = cosine_similarity(embeddings)

    # 计算所有物品对的平均差异度（1 - 相似度）
    n = len(items)
    total_dissimilarity = 0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dissimilarity = 1 - similarity_matrix[i, j]
            total_dissimilarity += dissimilarity
            count += 1

    return total_dissimilarity / count if count > 0 else 0


def calculate_category_coverage(items):
    """计算类别覆盖度"""
    categories = set(item.category for item in items)
    return len(categories)


def calculate_relevance(items):
    """计算平均相关性（精排分数）"""
    return np.mean([item.score for item in items])


def calculate_ndcg(items, ideal_items, k=10):
    """计算 NDCG@K"""
    # DCG
    dcg = sum(item.score / np.log2(i + 2) for i, item in enumerate(items[:k]))

    # IDCG
    idcg = sum(item.score / np.log2(i + 2) for i, item in enumerate(ideal_items[:k]))

    return dcg / idcg if idcg > 0 else 0


# ============ 4. 对比实验 ============

def compare_reranking_methods(items, top_k=20):
    """对比不同重排方法"""
    print("\n" + "=" * 60)
    print("重排方法对比")
    print("=" * 60)

    # 精排结果（基准）
    baseline = items[:top_k]

    # MMR
    mmr_reranker = MMRReranker(lambda_param=0.5)
    mmr_result = mmr_reranker.rerank(items, top_k)

    # 规则打散
    rule_reranker = RuleBasedReranker()
    rule_result = rule_reranker.rerank(items, top_k)

    # DPP
    dpp_reranker = DPPReranker()
    dpp_result = dpp_reranker.rerank(items, top_k)

    results = {
        '精排（基准）': baseline,
        'MMR': mmr_result,
        '规则打散': rule_result,
        'DPP': dpp_result
    }

    # 评估
    for name, result in results.items():
        diversity = calculate_diversity(result)
        category_coverage = calculate_category_coverage(result)
        relevance = calculate_relevance(result)
        ndcg = calculate_ndcg(result, baseline, k=10)

        print(f"\n{name}:")
        print(f"  多样性 (ILD): {diversity:.4f}")
        print(f"  类别覆盖度: {category_coverage}")
        print(f"  平均相关性: {relevance:.4f}")
        print(f"  NDCG@10: {ndcg:.4f}")

        # 展示前5个物品
        print(f"  前5个物品:")
        for i, item in enumerate(result[:5]):
            print(f"    {i+1}. ID={item.item_id}, 分数={item.score:.3f}, 类别={item.category}, 价格={item.price:.1f}")

    return results


# ============ 5. 可视化 ============

def visualize_reranking(results):
    """可视化重排效果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (name, items) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]

        # 类别分布
        categories = [item.category for item in items]
        positions = list(range(len(categories)))

        ax.scatter(positions, categories, s=100, alpha=0.6)
        ax.set_xlabel('Position in List')
        ax.set_ylabel('Category ID')
        ax.set_title(f'{name} - Category Distribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reranking_visualization.png', dpi=150)
    print("\n可视化已保存到 reranking_visualization.png")
    plt.close()


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  重排（Re-Ranking）- PyTorch实现")
    print("  优化推荐列表的多样性和用户体验")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("生成精排结果")
    print("=" * 60)

    # 生成精排后的物品列表
    items = generate_ranked_items(num_items=50, num_categories=10, embedding_dim=16)

    print(f"生成 {len(items)} 个物品")
    print(f"精排分数范围: [{min(item.score for item in items):.3f}, {max(item.score for item in items):.3f}]")

    # 对比重排方法
    results = compare_reranking_methods(items, top_k=20)

    # 可视化
    visualize_reranking(results)

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. 重排的核心目标
   ✓ 从单点优化 → 列表优化
   ✓ 平衡相关性和多样性
   ✓ 优化整体用户体验

2. 三种重排方法
   ✓ MMR：最大边际相关性（贪心 + 相似度）
   ✓ 规则打散：类别打散、价格打散（简单有效）
   ✓ DPP：行列式点过程（理论最优，计算复杂）

3. MMR 核心公式
   score = λ × relevance - (1-λ) × max_similarity
   - λ 控制相关性和多样性的权重
   - 贪心选择：每次选 score 最高的

4. 评价指标
   ✓ 多样性：ILD、类别覆盖度
   ✓ 相关性：平均分数、NDCG
   ✓ 用户体验：满意度、停留时长

5. 工业实践
   ✓ MMR：最常用（平衡效果和性能）
   ✓ 规则打散：简单有效
   ✓ DPP：理论优美，但计算昂贵
   ✓ 混合策略：规则 + 算法

6. 典型场景
   ✓ 电商：类别打散、价格打散
   ✓ 视频：时长打散、热度打散
   ✓ 新闻：话题打散、时间打散

7. 下一步
   → 混排：广告穿插、运营位
   → A/B 测试：评估重排效果
    """)

    print("\n✅ 重排学习完成！")
    print("\n提示: 重排是提升用户体验的关键环节")


if __name__ == "__main__":
    main()
