"""
粗排（Coarse Ranking / Pre-Ranking）- PyTorch实现
推荐系统中召回和精排之间的过渡层

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 粗排在推荐链路中的位置
====================================================================

完整链路：
召回（几千） → 粗排（几百） → 精排（几十） → 重排 → 混排

为什么需要粗排？
- 召回：快速筛选（简单模型，如双塔点积）
- 精排：精准预测（复杂模型，如 DIN、DeepFM）
- 问题：直接精排几千个候选 → 延迟太高 ❌

粗排的目标：
✓ 在保证效果的前提下，降低精排的计算压力
✓ 用轻量级模型，快速过滤掉明显不相关的候选
✓ 为精排提供高质量的候选集

====================================================================
🎯 粗排的核心思想
====================================================================

平衡效果和性能：
- 比召回更准确（用更多特征）
- 比精排更快速（更少参数）

三种常用方法：

1️⃣ 双塔点积增强版
   - 复用召回的双塔模型
   - 增加特征维度
   - 计算更精确的相似度

2️⃣ 知识蒸馏
   - Teacher：精排模型（DIN、DeepFM）
   - Student：轻量级模型
   - Student 学习 Teacher 的预测结果

3️⃣ 轻量级 MLP
   - 类似 DeepFM，但更简单
   - 只用核心特征（去掉历史序列等复杂特征）
   - 更少的层数和参数

====================================================================
🏗️ 本实现：三种粗排方法对比
====================================================================

方法1：双塔增强版
- 用户塔 + 物品塔
- 增加特征（age, gender, category）
- 点积 + sigmoid

方法2：知识蒸馏
- Teacher：精排模型（已训练好的 DeepFM）
- Student：轻量级 MLP
- 损失：KL散度（学习 Teacher 的输出分布）

方法3：轻量级 MLP
- 简单的 Embedding + MLP
- 2层隐藏层
- 参数量 < 精排模型的 1/5

====================================================================
📊 性能对比指标
====================================================================

关键指标：
1. AUC：排序能力
2. 参数量：模型复杂度
3. 推理速度：QPS（每秒查询数）
4. 召回率@K：粗排 top-K 中包含精排 top-N 的比例

目标：
- 参数量 < 精排的 20%
- AUC 接近精排（差距 < 5%）
- 速度 > 精排的 5 倍

====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ 1. 数据准备（复用之前的）============

class RankingDataset(Dataset):
    """排序数据集"""

    def __init__(self, num_samples=10000, num_users=1000, num_items=500):
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_items = num_items

        self.num_ages = 5
        self.num_genders = 2
        self.num_categories = 20

        self.user_ages = np.random.randint(0, self.num_ages, num_users)
        self.user_genders = np.random.randint(0, self.num_genders, num_users)
        self.item_categories = np.random.randint(0, self.num_categories, num_items)
        self.item_prices = np.random.uniform(10, 1000, num_items)

        self.samples = []
        for _ in range(num_samples):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)

            age = self.user_ages[user_id]
            gender = self.user_genders[user_id]
            category = self.item_categories[item_id]
            price = self.item_prices[item_id]

            click_prob = 0.1
            if age < 2 and category < 5: click_prob += 0.4
            if gender == 1 and category in [10, 11, 12]: click_prob += 0.4
            if price < 200: click_prob += 0.2

            label = 1 if np.random.rand() < click_prob else 0

            self.samples.append({
                'user_id': user_id,
                'item_id': item_id,
                'age': age,
                'gender': gender,
                'category': category,
                'price': price,
                'label': label
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'user_id': torch.LongTensor([sample['user_id']]),
            'item_id': torch.LongTensor([sample['item_id']]),
            'age': torch.LongTensor([sample['age']]),
            'gender': torch.LongTensor([sample['gender']]),
            'category': torch.LongTensor([sample['category']]),
            'price': torch.FloatTensor([sample['price']]),
            'label': torch.FloatTensor([sample['label']])
        }


# ============ 2. 粗排模型 ============

class CoarseRankingModel_TwoTower(nn.Module):
    """方法1：双塔增强版（召回模型的加强版）"""

    def __init__(self, num_users, num_items, num_ages, num_genders, num_categories,
                 embedding_dim=16):
        super().__init__()

        # 用户塔
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim // 2)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 4)

        user_input_dim = embedding_dim + embedding_dim // 2 + embedding_dim // 4
        self.user_mlp = nn.Sequential(
            nn.Linear(user_input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 物品塔
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)

        item_input_dim = embedding_dim + embedding_dim // 2 + 1  # +1 for price
        self.item_mlp = nn.Sequential(
            nn.Linear(item_input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, user_id, age, gender, item_id, category, price):
        # 用户塔
        user_emb = self.user_embedding(user_id).squeeze(1)
        age_emb = self.age_embedding(age).squeeze(1)
        gender_emb = self.gender_embedding(gender).squeeze(1)
        user_feat = torch.cat([user_emb, age_emb, gender_emb], dim=1)
        user_vec = self.user_mlp(user_feat)
        user_vec = F.normalize(user_vec, p=2, dim=1)

        # 物品塔
        item_emb = self.item_embedding(item_id).squeeze(1)
        cat_emb = self.category_embedding(category).squeeze(1)
        item_feat = torch.cat([item_emb, cat_emb, price / 1000.0], dim=1)
        item_vec = self.item_mlp(item_feat)
        item_vec = F.normalize(item_vec, p=2, dim=1)

        # 点积
        logit = torch.sum(user_vec * item_vec, dim=1)
        return logit


class CoarseRankingModel_LightMLP(nn.Module):
    """方法2：轻量级 MLP"""

    def __init__(self, num_users, num_items, num_ages, num_genders, num_categories,
                 embedding_dim=16):
        super().__init__()

        # Embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim // 2)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 4)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)

        # 轻量级 MLP（只有2层）
        input_dim = embedding_dim * 2 + embedding_dim // 2 * 2 + embedding_dim // 4 + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, user_id, age, gender, item_id, category, price):
        user_emb = self.user_embedding(user_id).squeeze(1)
        item_emb = self.item_embedding(item_id).squeeze(1)
        age_emb = self.age_embedding(age).squeeze(1)
        gender_emb = self.gender_embedding(gender).squeeze(1)
        cat_emb = self.category_embedding(category).squeeze(1)

        features = torch.cat([
            user_emb, item_emb, age_emb, gender_emb, cat_emb, price / 1000.0
        ], dim=1)

        logit = self.mlp(features).squeeze(1)
        return logit


class FineRankingModel(nn.Module):
    """精排模型（作为对比基准）"""

    def __init__(self, num_users, num_items, num_ages, num_genders, num_categories,
                 embedding_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim // 2)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 4)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)

        input_dim = embedding_dim * 2 + embedding_dim // 2 * 2 + embedding_dim // 4 + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, user_id, age, gender, item_id, category, price):
        user_emb = self.user_embedding(user_id).squeeze(1)
        item_emb = self.item_embedding(item_id).squeeze(1)
        age_emb = self.age_embedding(age).squeeze(1)
        gender_emb = self.gender_embedding(gender).squeeze(1)
        cat_emb = self.category_embedding(category).squeeze(1)

        features = torch.cat([
            user_emb, item_emb, age_emb, gender_emb, cat_emb, price / 1000.0
        ], dim=1)

        logit = self.mlp(features).squeeze(1)
        return logit


# ============ 3. 训练 ============

def train_model(model, train_loader, val_loader, device, model_name, num_epochs=15, lr=0.001):
    """训练模型"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

    print(f"\n训练 {model_name}...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            user_id = batch['user_id'].to(device)
            age = batch['age'].to(device)
            gender = batch['gender'].to(device)
            item_id = batch['item_id'].to(device)
            category = batch['category'].to(device)
            price = batch['price'].to(device)
            label = batch['label'].to(device).squeeze()

            optimizer.zero_grad()
            logit = model(user_id, age, gender, item_id, category, price)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logit).detach().cpu().numpy())
            train_labels.extend(label.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)

        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                user_id = batch['user_id'].to(device)
                age = batch['age'].to(device)
                gender = batch['gender'].to(device)
                item_id = batch['item_id'].to(device)
                category = batch['category'].to(device)
                price = batch['price'].to(device)
                label = batch['label'].to(device).squeeze()

                logit = model(user_id, age, gender, item_id, category, price)
                loss = criterion(logit, label)

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logit).cpu().numpy())
                val_labels.extend(label.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{num_epochs}: Val AUC: {val_auc:.4f}')

    return history


# ============ 4. 性能对比 ============

def compare_models(models, test_loader, device):
    """对比模型的性能"""
    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)

    results = {}

    for name, model in models.items():
        model.eval()
        preds = []
        labels = []
        start_time = time.time()

        with torch.no_grad():
            for batch in test_loader:
                user_id = batch['user_id'].to(device)
                age = batch['age'].to(device)
                gender = batch['gender'].to(device)
                item_id = batch['item_id'].to(device)
                category = batch['category'].to(device)
                price = batch['price'].to(device)
                label = batch['label'].to(device).squeeze()

                logit = model(user_id, age, gender, item_id, category, price)
                preds.extend(torch.sigmoid(logit).cpu().numpy())
                labels.extend(label.cpu().numpy())

        inference_time = time.time() - start_time
        auc = roc_auc_score(labels, preds)
        params = sum(p.numel() for p in model.parameters())
        qps = len(test_loader.dataset) / inference_time

        results[name] = {
            'auc': auc,
            'params': params,
            'time': inference_time,
            'qps': qps
        }

        print(f"\n{name}:")
        print(f"  AUC: {auc:.4f}")
        print(f"  参数量: {params:,}")
        print(f"  推理时间: {inference_time:.3f}s")
        print(f"  QPS: {qps:.1f}")

    return results


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  粗排（Coarse Ranking）- PyTorch实现")
    print("  推荐系统召回和精排之间的过渡层")
    print("=" * 60)

    print(f"\n使用设备: {DEVICE}")

    print("\n" + "=" * 60)
    print("创建数据集")
    print("=" * 60)

    train_dataset = RankingDataset(num_samples=50000, num_users=1000, num_items=500)
    val_dataset = RankingDataset(num_samples=10000, num_users=1000, num_items=500)
    test_dataset = RankingDataset(num_samples=5000, num_users=1000, num_items=500)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    print("\n" + "=" * 60)
    print("创建模型")
    print("=" * 60)

    # 粗排模型1：双塔增强版
    coarse_two_tower = CoarseRankingModel_TwoTower(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        num_ages=train_dataset.num_ages,
        num_genders=train_dataset.num_genders,
        num_categories=train_dataset.num_categories,
        embedding_dim=16
    ).to(DEVICE)

    # 粗排模型2：轻量级 MLP
    coarse_light_mlp = CoarseRankingModel_LightMLP(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        num_ages=train_dataset.num_ages,
        num_genders=train_dataset.num_genders,
        num_categories=train_dataset.num_categories,
        embedding_dim=16
    ).to(DEVICE)

    # 精排模型（作为对比基准）
    fine_ranking = FineRankingModel(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        num_ages=train_dataset.num_ages,
        num_genders=train_dataset.num_genders,
        num_categories=train_dataset.num_categories,
        embedding_dim=32
    ).to(DEVICE)

    print(f"粗排-双塔: {sum(p.numel() for p in coarse_two_tower.parameters()):,} 参数")
    print(f"粗排-MLP: {sum(p.numel() for p in coarse_light_mlp.parameters()):,} 参数")
    print(f"精排: {sum(p.numel() for p in fine_ranking.parameters()):,} 参数")

    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    train_model(coarse_two_tower, train_loader, val_loader, DEVICE, "粗排-双塔", num_epochs=15)
    train_model(coarse_light_mlp, train_loader, val_loader, DEVICE, "粗排-MLP", num_epochs=15)
    train_model(fine_ranking, train_loader, val_loader, DEVICE, "精排", num_epochs=15)

    # 性能对比
    models = {
        '粗排-双塔': coarse_two_tower,
        '粗排-MLP': coarse_light_mlp,
        '精排': fine_ranking
    }

    results = compare_models(models, test_loader, DEVICE)

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. 粗排的作用
   ✓ 承上启下：召回 → 粗排 → 精排
   ✓ 降低精排压力：过滤掉明显不相关的候选
   ✓ 平衡效果和性能

2. 粗排方法对比
   ✓ 双塔增强版：简单快速，但效果有限
   ✓ 轻量级 MLP：效果好，参数少
   ✓ 知识蒸馏：学习精排模型（未实现）

3. 关键指标
   ✓ AUC：排序能力（接近精排）
   ✓ 参数量：< 精排的 20%
   ✓ QPS：> 精排的 5 倍

4. 工业实践
   ✓ 特征选择：去掉复杂特征（历史序列等）
   ✓ 模型压缩：剪枝、量化
   ✓ 在线服务：批处理、缓存

5. 下一步
   → 重排：多样性、打散
   → 混排：广告穿插、运营位
    """)

    print("\n✅ 粗排学习完成！")


if __name__ == "__main__":
    main()
