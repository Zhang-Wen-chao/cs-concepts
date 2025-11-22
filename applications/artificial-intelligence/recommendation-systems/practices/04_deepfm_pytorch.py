"""
DeepFM 模型 - PyTorch实现
自动学习特征交叉的 CTR 预测模型（华为 2017）

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 从 Wide & Deep 到 DeepFM
====================================================================

Wide & Deep 的问题：
- Wide 部分需要人工设计交叉特征
- 如：age×gender, category×price, hour×device
- 工作量大，需要领域专家

DeepFM 的解决方案：
✓ 用 FM (Factorization Machines) 替代 Wide
✓ FM 自动学习所有特征的两两交叉（二阶交叉）
✓ 无需人工特征工程

====================================================================
🎯 什么是 FM (Factorization Machines)？
====================================================================

线性模型的局限：
y = w₁x₁ + w₂x₂ + w₃x₃ + b

只能学习特征的独立贡献，无法建模特征交叉：
- 无法学到 "年轻" + "女性" 的组合效应
- 无法学到 "电子产品" + "低价" 的交互

二阶多项式（暴力交叉）：
y = w₁x₁ + w₂x₂ + ... + w₁₂x₁x₂ + w₁₃x₁x₃ + w₂₃x₂x₃ + ...
                        ↑ 这些是交叉项

问题：n 个特征需要 n(n-1)/2 个交叉权重，参数爆炸！

FM 的优雅解决方案：
不直接学习 w_ij（交叉权重），而是学习每个特征的向量 v_i

交叉权重 = v_i · v_j（两个向量的点积）

y = 线性部分 + 交叉部分
  = Σ w_i x_i  +  Σ Σ (v_i · v_j) x_i x_j
                  i<j

====================================================================
🏗️ DeepFM 架构
====================================================================

                输入特征 (user, item, context)
                          ↓
                    Embedding 层
                   /      |      \
                  /       |       \
            ┌─────┐   ┌─────┐   ┌─────┐
            │  FM │   │Dense│   │Deep │
            │部分 │   │部分 │   │部分 │
            └─────┘   └─────┘   └─────┘
                \       |       /
                 \      |      /
                  \     |     /
                   ┌─────────┐
                   │ 加权求和 │
                   └─────────┘
                        ↓
                   sigmoid(logit)
                        ↓
                   点击概率 (0-1)

详细结构：

输入: [user_id, age, gender, item_id, category, price, ...]
  ↓
Embedding: 每个特征 → k 维向量
  [v_user, v_age, v_gender, v_item, v_category, ...]
  ↓
┌─────────────────────────────────────────┐
│ 1️⃣ Dense 部分（一阶）                   │
│   y₁ = Σ w_i x_i                        │
│   每个特征的独立贡献                     │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 2️⃣ FM 部分（二阶交叉）                  │
│   y₂ = Σ Σ (v_i · v_j) x_i x_j         │
│       i<j                                │
│   所有特征两两交叉（自动！）              │
│                                          │
│   高效计算技巧：                         │
│   Σ Σ (v_i·v_j)x_i x_j                 │
│   = 0.5 * [Σ(Σv_i x_i)² - Σ(v_i²x_i²)] │
│                                          │
│   复杂度: O(n²k) → O(nk)                │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ 3️⃣ Deep 部分（高阶交叉）                │
│   [v_1, v_2, ..., v_n] 拼接             │
│        ↓                                 │
│   MLP(256 → 128 → 64)                   │
│        ↓                                 │
│   y₃ = 高阶非线性组合                    │
└─────────────────────────────────────────┘
  ↓
输出: logit = y₁ + y₂ + y₃

====================================================================
🔑 FM 的高效计算技巧
====================================================================

原始公式（复杂度 O(n²k)）：
Σ Σ (v_i · v_j) x_i x_j
i<j

展开 v_i · v_j：
Σ Σ (Σ v_i,f × v_j,f) x_i x_j
i<j  f

调整求和顺序：
Σ [Σ Σ v_i,f × v_j,f × x_i × x_j]
f  i<j

数学变换（关键！）：
Σ Σ v_i,f × v_j,f × x_i × x_j
i<j

= 0.5 × [Σ Σ v_i,f v_j,f x_i x_j - Σ v_i,f² x_i²]
        i,j              i

= 0.5 × [(Σ v_i,f x_i)² - Σ v_i,f² x_i²]
         i               i

最终（复杂度 O(nk)）：
FM = 0.5 × Σ [(Σ v_i,f x_i)² - Σ v_i,f² x_i²]
           f   i              i

====================================================================
💡 DeepFM vs Wide & Deep
====================================================================

| 维度           | Wide & Deep        | DeepFM            |
|----------------|-------------------|-------------------|
| 浅层部分       | Wide (人工交叉)    | FM (自动交叉)     |
| 深层部分       | Deep (MLP)        | Deep (MLP)        |
| 特征工程       | 需要人工设计       | 自动学习          |
| 训练复杂度     | 中等              | 较低              |
| 工业应用       | Google, 需要专家   | 华为, 更通用      |

====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ 1. 数据准备（复用 Wide & Deep 的数据）============

class CTRDataset(Dataset):
    """点击率预测数据集（与 Wide & Deep 相同）"""

    def __init__(self, num_samples=10000, num_users=1000, num_items=500):
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_items = num_items

        self.num_ages = 5
        self.num_genders = 2
        self.num_cities = 10
        self.num_categories = 20
        self.num_brands = 50
        self.num_hours = 24
        self.num_devices = 2

        self.user_ages = np.random.randint(0, self.num_ages, num_users)
        self.user_genders = np.random.randint(0, self.num_genders, num_users)
        self.user_cities = np.random.randint(0, self.num_cities, num_users)

        self.item_categories = np.random.randint(0, self.num_categories, num_items)
        self.item_brands = np.random.randint(0, self.num_brands, num_items)
        self.item_prices = np.random.uniform(10, 1000, num_items)

        self.samples = []
        for _ in range(num_samples):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)
            hour = np.random.randint(0, self.num_hours)
            device = np.random.randint(0, self.num_devices)

            age = self.user_ages[user_id]
            gender = self.user_genders[user_id]
            category = self.item_categories[item_id]
            price = self.item_prices[item_id]

            click_prob = 0.1
            if age < 2 and category < 5: click_prob += 0.4
            if gender == 1 and category in [10, 11, 12]: click_prob += 0.4
            if hour > 18 and category in [15, 16, 17]: click_prob += 0.3
            if price < 100: click_prob += 0.2

            label = 1 if np.random.rand() < click_prob else 0

            self.samples.append({
                'user_id': user_id,
                'item_id': item_id,
                'age': self.user_ages[user_id],
                'gender': self.user_genders[user_id],
                'city': self.user_cities[user_id],
                'category': category,
                'brand': self.item_brands[item_id],
                'price': price,
                'hour': hour,
                'device': device,
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
            'city': torch.LongTensor([sample['city']]),
            'category': torch.LongTensor([sample['category']]),
            'brand': torch.LongTensor([sample['brand']]),
            'hour': torch.LongTensor([sample['hour']]),
            'device': torch.LongTensor([sample['device']]),
            'price': torch.FloatTensor([sample['price']]),
            'label': torch.FloatTensor([sample['label']])
        }


# ============ 2. DeepFM 模型 ============

class DenseLayer(nn.Module):
    """Dense 部分：一阶线性部分"""

    def __init__(self, num_features):
        super().__init__()
        # 每个特征一个权重
        self.fc = nn.Linear(num_features, 1, bias=True)

    def forward(self, x):
        """
        参数:
            x: (batch, num_features) - 特征向量（0/1 或归一化后的值）

        返回:
            output: (batch, 1) - 一阶输出
        """
        return self.fc(x)


class FMLayer(nn.Module):
    """FM 部分：二阶特征交叉"""

    def __init__(self, num_features, embedding_dim):
        super().__init__()
        # 为每个特征学习一个 k 维向量
        self.embedding = nn.Parameter(torch.randn(num_features, embedding_dim))
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, x):
        """
        参数:
            x: (batch, num_features) - 特征向量

        返回:
            output: (batch, 1) - 二阶交叉输出

        高效计算：
        Σ Σ (v_i · v_j) x_i x_j
        = 0.5 * [Σ(Σv_i x_i)² - Σ(v_i² x_i²)]
        """
        # x: (batch, num_features)
        # embedding: (num_features, k)

        # 1. Σ v_i x_i (对每个维度)
        # x.unsqueeze(2): (batch, num_features, 1)
        # embedding: (num_features, k)
        square_of_sum = torch.pow(torch.matmul(x, self.embedding), 2)  # (batch, k)

        # 2. Σ v_i² x_i²
        sum_of_square = torch.matmul(torch.pow(x, 2), torch.pow(self.embedding, 2))  # (batch, k)

        # 3. 0.5 * (square_of_sum - sum_of_square)
        fm_output = 0.5 * (square_of_sum - sum_of_square)  # (batch, k)

        # 4. 对 k 维求和
        fm_output = torch.sum(fm_output, dim=1, keepdim=True)  # (batch, 1)

        return fm_output


class DeepFMModel(nn.Module):
    """DeepFM 完整模型"""

    def __init__(self,
                 num_users, num_items, num_ages, num_genders, num_cities,
                 num_categories, num_brands, num_hours, num_devices,
                 embedding_dim=16, hidden_dims=[256, 128, 64]):
        super().__init__()

        # ============ Embedding 层（共享）============
        # 这些 Embedding 会被 FM 和 Deep 共享

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim // 2)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 4)
        self.city_embedding = nn.Embedding(num_cities, embedding_dim // 2)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)
        self.brand_embedding = nn.Embedding(num_brands, embedding_dim // 2)
        self.hour_embedding = nn.Embedding(num_hours, embedding_dim // 4)
        self.device_embedding = nn.Embedding(num_devices, embedding_dim // 4)

        # 计算特征总数（用于 Dense 和 FM）
        # 这里简化：只考虑离散特征的 one-hot 维度
        self.num_features = (num_users + num_items + num_ages + num_genders +
                            num_cities + num_categories + num_brands +
                            num_hours + num_devices + 1)  # +1 for price

        # ============ 1. Dense 部分（一阶）============
        self.dense = DenseLayer(self.num_features)

        # ============ 2. FM 部分（二阶）============
        self.fm = FMLayer(self.num_features, embedding_dim)

        # ============ 3. Deep 部分（高阶）============
        deep_input_dim = (
            embedding_dim * 2 +
            (embedding_dim // 2) * 4 +
            (embedding_dim // 4) * 3 +
            1  # price
        )

        layers = []
        input_dim = deep_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.deep_mlp = nn.Sequential(*layers)
        self.deep_output = nn.Linear(hidden_dims[-1], 1)

    def _create_feature_vector(self, user_id, item_id, age, gender, city,
                               category, brand, hour, device, price):
        """
        创建 one-hot 风格的特征向量（用于 Dense 和 FM）

        实际生产中会用稀疏矩阵，这里为了简化用稠密向量
        """
        batch_size = user_id.size(0)
        features = torch.zeros(batch_size, self.num_features).to(user_id.device)

        # 将每个特征的 ID 转换为 one-hot 的位置
        # 这里简化处理：直接用归一化后的 ID
        offset = 0

        features[:, offset + user_id.squeeze(1)] = 1
        offset += self.user_embedding.num_embeddings

        features[:, offset + item_id.squeeze(1)] = 1
        offset += self.item_embedding.num_embeddings

        features[:, offset + age.squeeze(1)] = 1
        offset += self.age_embedding.num_embeddings

        features[:, offset + gender.squeeze(1)] = 1
        offset += self.gender_embedding.num_embeddings

        features[:, offset + city.squeeze(1)] = 1
        offset += self.city_embedding.num_embeddings

        features[:, offset + category.squeeze(1)] = 1
        offset += self.category_embedding.num_embeddings

        features[:, offset + brand.squeeze(1)] = 1
        offset += self.brand_embedding.num_embeddings

        features[:, offset + hour.squeeze(1)] = 1
        offset += self.hour_embedding.num_embeddings

        features[:, offset + device.squeeze(1)] = 1
        offset += self.device_embedding.num_embeddings

        # 数值特征归一化
        features[:, -1] = price.squeeze() / 1000.0

        return features

    def forward(self, user_id, item_id, age, gender, city, category, brand,
                price, hour, device):
        """
        前向传播

        返回:
            logit: (batch,) - Dense + FM + Deep 的组合输出
        """
        batch_size = user_id.size(0)

        # ============ 1. Dense 部分 ============
        features = self._create_feature_vector(
            user_id, item_id, age, gender, city, category, brand, hour, device, price
        )
        logit_dense = self.dense(features).squeeze(1)  # (batch,)

        # ============ 2. FM 部分 ============
        logit_fm = self.fm(features).squeeze(1)  # (batch,)

        # ============ 3. Deep 部分 ============
        user_emb = self.user_embedding(user_id).squeeze(1)
        item_emb = self.item_embedding(item_id).squeeze(1)
        age_emb = self.age_embedding(age).squeeze(1)
        gender_emb = self.gender_embedding(gender).squeeze(1)
        city_emb = self.city_embedding(city).squeeze(1)
        category_emb = self.category_embedding(category).squeeze(1)
        brand_emb = self.brand_embedding(brand).squeeze(1)
        hour_emb = self.hour_embedding(hour).squeeze(1)
        device_emb = self.device_embedding(device).squeeze(1)

        deep_input = torch.cat([
            user_emb, item_emb, age_emb, gender_emb, city_emb,
            category_emb, brand_emb, hour_emb, device_emb, price
        ], dim=1)

        deep_hidden = self.deep_mlp(deep_input)
        logit_deep = self.deep_output(deep_hidden).squeeze(1)  # (batch,)

        # ============ 组合输出 ============
        logit = logit_dense + logit_fm + logit_deep

        return logit


# ============ 3. 训练 ============

def train_model(model, train_loader, val_loader, device, num_epochs=30, lr=0.001):
    """训练 DeepFM 模型"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

    print("\n开始训练...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            user_id = batch['user_id'].to(device)
            item_id = batch['item_id'].to(device)
            age = batch['age'].to(device)
            gender = batch['gender'].to(device)
            city = batch['city'].to(device)
            category = batch['category'].to(device)
            brand = batch['brand'].to(device)
            price = batch['price'].to(device)
            hour = batch['hour'].to(device)
            device_feat = batch['device'].to(device)
            label = batch['label'].to(device).squeeze()

            optimizer.zero_grad()

            logit = model(user_id, item_id, age, gender, city, category,
                         brand, price, hour, device_feat)
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
                item_id = batch['item_id'].to(device)
                age = batch['age'].to(device)
                gender = batch['gender'].to(device)
                city = batch['city'].to(device)
                category = batch['category'].to(device)
                brand = batch['brand'].to(device)
                price = batch['price'].to(device)
                hour = batch['hour'].to(device)
                device_feat = batch['device'].to(device)
                label = batch['label'].to(device).squeeze()

                logit = model(user_id, item_id, age, gender, city, category,
                            brand, price, hour, device_feat)
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
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

    return history


# ============ 4. 可视化 ============

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

    ax2.plot(history['train_auc'], label='Train AUC')
    ax2.plot(history['val_auc'], label='Val AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Training and Validation AUC')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('deepfm_training.png', dpi=150)
    print("\n训练历史已保存到 deepfm_training.png")
    plt.close()


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  DeepFM 模型 - PyTorch实现")
    print("  自动学习特征交叉的 CTR 预测模型")
    print("=" * 60)

    print(f"\n使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("创建数据集")
    print("=" * 60)

    train_dataset = CTRDataset(num_samples=20000, num_users=1000, num_items=500)
    val_dataset = CTRDataset(num_samples=5000, num_users=1000, num_items=500)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    train_labels = [s['label'] for s in train_dataset.samples]
    pos_ratio = sum(train_labels) / len(train_labels)
    print(f"正样本比例: {pos_ratio:.2%}")

    print("\n" + "=" * 60)
    print("创建 DeepFM 模型")
    print("=" * 60)

    model = DeepFMModel(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        num_ages=train_dataset.num_ages,
        num_genders=train_dataset.num_genders,
        num_cities=train_dataset.num_cities,
        num_categories=train_dataset.num_categories,
        num_brands=train_dataset.num_brands,
        num_hours=train_dataset.num_hours,
        num_devices=train_dataset.num_devices,
        embedding_dim=16,
        hidden_dims=[256, 128, 64]
    ).to(DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    history = train_model(model, train_loader, val_loader, DEVICE, num_epochs=30, lr=0.001)

    plot_training_history(history)

    print("\n" + "=" * 60)
    print("测试预测")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for i in range(3):
            sample = val_dataset[i]

            user_id = sample['user_id'].unsqueeze(0).to(DEVICE)
            item_id = sample['item_id'].unsqueeze(0).to(DEVICE)
            age = sample['age'].unsqueeze(0).to(DEVICE)
            gender = sample['gender'].unsqueeze(0).to(DEVICE)
            city = sample['city'].unsqueeze(0).to(DEVICE)
            category = sample['category'].unsqueeze(0).to(DEVICE)
            brand = sample['brand'].unsqueeze(0).to(DEVICE)
            price = sample['price'].unsqueeze(0).to(DEVICE)
            hour = sample['hour'].unsqueeze(0).to(DEVICE)
            device_feat = sample['device'].unsqueeze(0).to(DEVICE)

            logit = model(user_id, item_id, age, gender, city, category,
                         brand, price, hour, device_feat)
            pred_prob = torch.sigmoid(logit).item()
            true_label = sample['label'].item()

            print(f"\n样本 {i+1}:")
            print(f"  用户ID: {user_id.item()}, 物品ID: {item_id.item()}")
            print(f"  特征: age={age.item()}, gender={gender.item()}, category={category.item()}, price={price.item():.1f}")
            print(f"  预测概率: {pred_prob:.3f}")
            print(f"  真实标签: {int(true_label)}")
            print(f"  预测结果: {'点击 ✓' if pred_prob > 0.5 else '不点击 ✗'}")

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. DeepFM 架构
   ✓ Dense: 一阶线性部分（y = Σ w_i x_i）
   ✓ FM: 二阶交叉部分（自动学习所有两两交叉）
   ✓ Deep: 高阶非线性部分（MLP）
   ✓ 组合: logit = logit_dense + logit_fm + logit_deep

2. FM (Factorization Machines) 核心
   ✓ 不直接学习交叉权重 w_ij
   ✓ 学习特征向量 v_i，交叉权重 = v_i · v_j
   ✓ 高效计算: O(n²k) → O(nk)
   ✓ 自动学习所有特征的两两交叉（无需人工设计）

3. DeepFM vs Wide & Deep
   Wide & Deep:
   - Wide: 需要人工设计交叉特征
   - 工作量大，需要领域知识

   DeepFM:
   - FM: 自动学习二阶交叉
   - 无需特征工程，更通用

4. 共享 Embedding
   ✓ FM 和 Deep 共享同一套 Embedding
   ✓ 端到端训练，特征表示更一致
   ✓ 参数更少，训练更快

5. 工业应用
   ✓ 华为应用商店推荐
   ✓ 适合特征多、交叉复杂的场景
   ✓ 相比 Wide & Deep 更易部署（无需人工特征工程）

6. 下一步
   → DIN (Deep Interest Network): 用户兴趣建模
   → DIEN: 用户兴趣进化网络
   → 多任务学习: 同时预测点击和转化
    """)

    print("\n✅ DeepFM 学习完成！")
    print("\n提示: DeepFM 通过 FM 自动化了特征交叉，是工业界广泛使用的模型")


if __name__ == "__main__":
    main()
