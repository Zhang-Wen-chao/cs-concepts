"""
双塔模型 (Two-Tower Model) - PyTorch实现
推荐系统召回阶段的核心深度学习架构

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 推荐系统完整链路
====================================================================

用户请求
   ↓
1. 召回（Retrieval）← 我们在这里！
   - 从百万级物品中快速筛选出几千个候选
   - 双塔模型：用户塔 + 物品塔 → 向量相似度
   ↓
2. 粗排（Pre-Ranking，可选）
   - 几千个 → 几百个
   ↓
3. 精排（Ranking）
   - 几百个 → 几十个
   - Wide & Deep, DeepFM, DIN 等
   ↓
4. 重排（Re-Ranking）
   - 多样性、打散
   ↓
5. 混排
   - 插入广告、运营内容
   ↓
展示给用户

====================================================================
🎯 为什么需要双塔模型？
====================================================================

传统召回问题：
- 协同过滤：无法利用丰富的特征（年龄、性别、类别等）
- 矩阵分解：只能处理 user_id 和 item_id

双塔模型优势：
✓ 可以使用任意特征（ID、类别、文本、图像）
✓ 训练和推理分离（离线向量化 + 在线 ANN 检索）
✓ 可扩展到百万级用户和物品

====================================================================
🏗️ 双塔模型架构
====================================================================

训练阶段：
    用户特征                    物品特征
    [user_id,                  [item_id,
     age,                       category,
     gender,          ×         price,
     history...]                tags...]
        ↓                          ↓
    用户塔(MLP)              物品塔(MLP)
    512→256→128              256→128
        ↓                          ↓
    用户向量(128维)  ─────×─────  物品向量(128维)
                         ↓
                    余弦相似度
                         ↓
                      分数 (0-1)
                         ↓
                    交叉熵损失

推理阶段（两步走）：
    1. 离线：对所有物品生成向量，存入向量数据库
    2. 在线：
       - 用户塔生成用户向量
       - ANN 检索最相似的 top-K 物品向量
       - 返回对应的物品ID

====================================================================
⚠️ 核心问题：mask 应该在哪里？
====================================================================

❌ 错误位置：在预测分数计算中
    pred = torch.sum(logits * mask, dim=-1)  # 错误！

✓ 正确位置：在损失函数中
    loss = criterion(pred, label) * mask     # 正确！
    loss = loss.sum() / mask.sum()

原因：
1. mask 是训练技巧（样本加权），不是模型逻辑
2. 推理时没有 mask，如果 mask 在 pred 里会导致不一致
3. mask 只影响梯度传播，不应该改变模型输出

====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ 1. 数据准备（模拟 MovieLens 风格）============

class MovieLensDataset(Dataset):
    """模拟 MovieLens 数据集

    用户特征：user_id, age_group, gender
    物品特征：item_id, category, year
    交互：rating (1-5)，我们将 >=4 的视为正样本
    """

    def __init__(self, num_users=1000, num_items=500, num_samples=10000):
        self.num_users = num_users
        self.num_items = num_items

        # 生成用户特征
        self.user_ages = np.random.randint(0, 5, num_users)  # 5个年龄段
        self.user_genders = np.random.randint(0, 2, num_users)  # 2种性别

        # 生成物品特征
        self.item_categories = np.random.randint(0, 10, num_items)  # 10个类别
        self.item_years = np.random.randint(0, 5, num_items)  # 5个年代

        # 生成交互数据（用户-物品对 + 标签）
        self.samples = []
        for _ in range(num_samples):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)

            # 模拟相似度：同年龄段用户喜欢同类别物品（简化规则）
            user_age = self.user_ages[user_id]
            item_cat = self.item_categories[item_id]

            # 如果用户年龄段和物品类别匹配，更可能是正样本
            label = 1 if (user_age % 10 == item_cat % 10) and np.random.rand() > 0.3 else 0

            self.samples.append((user_id, item_id, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, item_id, label = self.samples[idx]

        # 用户特征
        user_age = self.user_ages[user_id]
        user_gender = self.user_genders[user_id]

        # 物品特征
        item_cat = self.item_categories[item_id]
        item_year = self.item_years[item_id]

        return {
            'user_id': torch.LongTensor([user_id]),
            'user_age': torch.LongTensor([user_age]),
            'user_gender': torch.LongTensor([user_gender]),
            'item_id': torch.LongTensor([item_id]),
            'item_cat': torch.LongTensor([item_cat]),
            'item_year': torch.LongTensor([item_year]),
            'label': torch.FloatTensor([label])
        }


# ============ 2. 双塔模型 ============

class UserTower(nn.Module):
    """用户塔：将用户特征映射为固定维度的向量"""

    def __init__(self, num_users, num_ages, num_genders, embedding_dim=32, hidden_dim=128):
        super().__init__()

        # Embedding 层（将离散特征映射为向量）
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim // 2)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 4)

        # MLP 层（多层感知机）
        input_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)  # 最终用户向量维度
        )

    def forward(self, user_id, user_age, user_gender):
        """
        参数:
            user_id: (batch_size, 1)
            user_age: (batch_size, 1)
            user_gender: (batch_size, 1)

        返回:
            user_vector: (batch_size, hidden_dim // 2) - 用户向量
        """
        # 1. Embedding
        user_emb = self.user_embedding(user_id).squeeze(1)  # (batch, embedding_dim)
        age_emb = self.age_embedding(user_age).squeeze(1)
        gender_emb = self.gender_embedding(user_gender).squeeze(1)

        # 2. 拼接所有特征
        x = torch.cat([user_emb, age_emb, gender_emb], dim=1)

        # 3. 通过 MLP 得到用户向量
        user_vector = self.mlp(x)

        # 4. L2 归一化（重要！保证向量在单位球面上，余弦相似度 = 点积）
        user_vector = F.normalize(user_vector, p=2, dim=1)

        return user_vector


class ItemTower(nn.Module):
    """物品塔：将物品特征映射为固定维度的向量"""

    def __init__(self, num_items, num_categories, num_years, embedding_dim=32, hidden_dim=128):
        super().__init__()

        # Embedding 层
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.cat_embedding = nn.Embedding(num_categories, embedding_dim // 2)
        self.year_embedding = nn.Embedding(num_years, embedding_dim // 4)

        # MLP 层
        input_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)  # 最终物品向量维度
        )

    def forward(self, item_id, item_cat, item_year):
        """
        参数:
            item_id: (batch_size, 1)
            item_cat: (batch_size, 1)
            item_year: (batch_size, 1)

        返回:
            item_vector: (batch_size, hidden_dim // 2) - 物品向量
        """
        item_emb = self.item_embedding(item_id).squeeze(1)
        cat_emb = self.cat_embedding(item_cat).squeeze(1)
        year_emb = self.year_embedding(item_year).squeeze(1)

        x = torch.cat([item_emb, cat_emb, year_emb], dim=1)
        item_vector = self.mlp(x)

        # L2 归一化
        item_vector = F.normalize(item_vector, p=2, dim=1)

        return item_vector


class TwoTowerModel(nn.Module):
    """完整的双塔模型"""

    def __init__(self, num_users, num_items, num_ages=5, num_genders=2,
                 num_categories=10, num_years=5, embedding_dim=32, hidden_dim=128):
        super().__init__()

        self.user_tower = UserTower(num_users, num_ages, num_genders, embedding_dim, hidden_dim)
        self.item_tower = ItemTower(num_items, num_categories, num_years, embedding_dim, hidden_dim)

    def forward(self, user_id, user_age, user_gender, item_id, item_cat, item_year):
        """
        训练时的前向传播

        返回:
            similarity: (batch_size,) - 用户-物品相似度分数 (0-1)
            user_vector: (batch_size, dim) - 用户向量（用于分析）
            item_vector: (batch_size, dim) - 物品向量（用于分析）
        """
        # 1. 生成用户向量和物品向量
        user_vector = self.user_tower(user_id, user_age, user_gender)
        item_vector = self.item_tower(item_id, item_cat, item_year)

        # 2. 计算相似度（点积，因为已经归一化，所以等价于余弦相似度）
        similarity = torch.sum(user_vector * item_vector, dim=1)  # (batch_size,)

        # 3. 映射到 (0, 1) 区间（使用 sigmoid）
        similarity = torch.sigmoid(similarity)

        return similarity, user_vector, item_vector

    def get_user_vector(self, user_id, user_age, user_gender):
        """推理时：只获取用户向量"""
        return self.user_tower(user_id, user_age, user_gender)

    def get_item_vector(self, item_id, item_cat, item_year):
        """推理时：只获取物品向量（用于离线向量化）"""
        return self.item_tower(item_id, item_cat, item_year)


# ============ 3. 训练 ============

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.001):
    """训练双塔模型"""
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n开始训练双塔模型...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            user_id = batch['user_id'].to(device)
            user_age = batch['user_age'].to(device)
            user_gender = batch['user_gender'].to(device)
            item_id = batch['item_id'].to(device)
            item_cat = batch['item_cat'].to(device)
            item_year = batch['item_year'].to(device)
            label = batch['label'].to(device).squeeze()

            optimizer.zero_grad()

            # 前向传播
            pred, _, _ = model(user_id, user_age, user_gender, item_id, item_cat, item_year)

            # 计算损失（注意：这里没有 mask！）
            loss = criterion(pred, label)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            train_correct += ((pred > 0.5) == label).sum().item()
            train_total += label.size(0)

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                user_id = batch['user_id'].to(device)
                user_age = batch['user_age'].to(device)
                user_gender = batch['user_gender'].to(device)
                item_id = batch['item_id'].to(device)
                item_cat = batch['item_cat'].to(device)
                item_year = batch['item_year'].to(device)
                label = batch['label'].to(device).squeeze()

                pred, _, _ = model(user_id, user_age, user_gender, item_id, item_cat, item_year)
                loss = criterion(pred, label)

                val_loss += loss.item()
                val_correct += ((pred > 0.5) == label).sum().item()
                val_total += label.size(0)

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return history


# ============ 4. 推理（召回）============

def build_item_index(model, dataset, device):
    """离线构建物品向量索引

    在实际生产中，这一步会：
    1. 对所有物品生成向量
    2. 存入向量数据库（Faiss, Milvus 等）
    3. 支持 ANN（近似最近邻）检索
    """
    model.eval()
    item_vectors = []
    item_ids = []

    print("\n构建物品向量索引...")
    with torch.no_grad():
        for item_id in range(dataset.num_items):
            item_cat = dataset.item_categories[item_id]
            item_year = dataset.item_years[item_id]

            # 转换为 tensor
            item_id_t = torch.LongTensor([[item_id]]).to(device)
            item_cat_t = torch.LongTensor([[item_cat]]).to(device)
            item_year_t = torch.LongTensor([[item_year]]).to(device)

            # 生成物品向量
            item_vec = model.get_item_vector(item_id_t, item_cat_t, item_year_t)

            item_vectors.append(item_vec.cpu().numpy())
            item_ids.append(item_id)

    item_vectors = np.vstack(item_vectors)  # (num_items, vector_dim)
    print(f"物品向量索引构建完成：{item_vectors.shape}")

    return item_vectors, item_ids


def recall_for_user(model, user_id, dataset, item_vectors, item_ids, device, top_k=10):
    """在线召回：为用户召回 top-K 物品

    在实际生产中：
    1. 实时生成用户向量
    2. 在向量数据库中检索最相似的 top-K 物品向量
    3. 返回物品 ID
    """
    model.eval()

    with torch.no_grad():
        # 获取用户特征
        user_age = dataset.user_ages[user_id]
        user_gender = dataset.user_genders[user_id]

        # 转换为 tensor
        user_id_t = torch.LongTensor([[user_id]]).to(device)
        user_age_t = torch.LongTensor([[user_age]]).to(device)
        user_gender_t = torch.LongTensor([[user_gender]]).to(device)

        # 生成用户向量
        user_vec = model.get_user_vector(user_id_t, user_age_t, user_gender_t)
        user_vec = user_vec.cpu().numpy()  # (1, vector_dim)

        # 计算与所有物品的相似度（暴力检索，实际用 ANN）
        similarities = np.dot(item_vectors, user_vec.T).squeeze()  # (num_items,)

        # 获取 top-K
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        top_k_items = [item_ids[i] for i in top_k_indices]
        top_k_scores = similarities[top_k_indices]

    return top_k_items, top_k_scores


# ============ 5. 可视化 ============

def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('two_tower_training.png', dpi=150)
    print("\n训练历史已保存到 two_tower_training.png")
    plt.close()


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  双塔模型 (Two-Tower Model) - PyTorch实现")
    print("  推荐系统召回阶段的核心架构")
    print("=" * 60)

    # 检查设备
    print(f"\n使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 创建数据集
    print("\n" + "=" * 60)
    print("创建模拟数据集")
    print("=" * 60)

    train_dataset = MovieLensDataset(num_users=1000, num_items=500, num_samples=20000)
    val_dataset = MovieLensDataset(num_users=1000, num_items=500, num_samples=5000)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"用户数: {train_dataset.num_users}")
    print(f"物品数: {train_dataset.num_items}")

    # 创建模型
    print("\n" + "=" * 60)
    print("创建双塔模型")
    print("=" * 60)

    model = TwoTowerModel(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        num_ages=5,
        num_genders=2,
        num_categories=10,
        num_years=5,
        embedding_dim=32,
        hidden_dim=128
    ).to(DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"用户向量维度: 64")
    print(f"物品向量维度: 64")

    # 训练模型
    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    history = train_model(model, train_loader, val_loader, DEVICE, num_epochs=30, lr=0.001)

    # 可视化
    plot_training_history(history)

    # 构建物品索引
    print("\n" + "=" * 60)
    print("构建物品向量索引（离线）")
    print("=" * 60)

    item_vectors, item_ids = build_item_index(model, train_dataset, DEVICE)

    # 测试召回
    print("\n" + "=" * 60)
    print("测试召回（在线）")
    print("=" * 60)

    test_user_ids = [0, 10, 100]
    for user_id in test_user_ids:
        top_items, scores = recall_for_user(model, user_id, train_dataset, item_vectors, item_ids, DEVICE, top_k=10)

        print(f"\n用户 {user_id} 的召回结果 (Top-10):")
        print(f"  用户特征: age={train_dataset.user_ages[user_id]}, gender={train_dataset.user_genders[user_id]}")
        print(f"  召回物品ID: {top_items}")
        print(f"  相似度分数: {[f'{s:.3f}' for s in scores]}")

    # 总结
    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. 双塔模型结构
   ✓ 用户塔: 用户特征 → 用户向量
   ✓ 物品塔: 物品特征 → 物品向量
   ✓ 相似度: 点积/余弦相似度

2. 训练与推理分离
   ✓ 训练: 同时计算用户和物品向量，优化相似度
   ✓ 推理:
     - 离线: 生成所有物品向量，存入向量数据库
     - 在线: 生成用户向量，ANN 检索 top-K

3. 关键技术点
   ✓ L2 归一化: 保证向量在单位球面上
   ✓ 余弦相似度 = 归一化后的点积
   ✓ Embedding: 将离散特征映射为连续向量
   ✓ MLP: 学习特征的非线性组合

4. mask 的正确位置 ⚠️
   ✓ 训练时: 在损失函数中使用 mask（样本加权）
   ✗ 不要在模型输出中使用 mask！

   原因:
   - mask 是训练技巧，不是模型逻辑
   - 推理时没有 mask，会导致不一致
   - mask 只影响梯度，不应改变输出

5. 工业实践
   ✓ 向量数据库: Faiss, Milvus, Elasticsearch
   ✓ ANN 检索: HNSW, IVF, Product Quantization
   ✓ 特征工程: ID、类别、统计、序列特征
   ✓ 负采样: 随机负样本、难负样本

6. 下一步
   → Wide & Deep (精排模型)
   → DeepFM (特征交叉)
   → DIN (注意力机制)
   → 多任务学习 (MMoE)
    """)

    print("\n✅ 双塔模型学习完成！")
    print("\n提示: 双塔模型是召回阶段的基础，接下来可以学习精排模型")


if __name__ == "__main__":
    main()
