"""
多任务学习 (Multi-Task Learning) - PyTorch实现
同时预测点击和转化的推荐系统模型

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 为什么需要多任务学习？
====================================================================

单任务模型的问题：
- 只优化 CTR（点击率）
- 点击 ≠ 转化（购买/下单）
- 很多点击是"看看而已"，对业务没价值

真实场景：
用户行为链路：曝光 → 点击 → 转化（购买/下单/注册）

业务目标：
- 不只是要点击，更要转化！
- GMV（成交额）= 点击量 × 转化率 × 客单价

====================================================================
🎯 多任务学习的核心思想
====================================================================

同时预测多个相关任务：
1. CTR（Click-Through Rate）：点击概率
2. CVR（Conversion Rate）：转化概率

优势：
✓ 任务之间共享特征表示
✓ 互相辅助学习（点击和转化有相关性）
✓ 缓解数据稀疏（转化样本少，可以借助点击样本）

====================================================================
🏗️ 多任务学习架构（Shared-Bottom）
====================================================================

最简单的架构：

         输入特征 (user + item)
                ↓
          共享 Embedding 层
                ↓
          共享 MLP 层（底层）
         ┌──────┴──────┐
         ↓             ↓
    CTR Tower     CVR Tower
    (专用层)      (专用层)
         ↓             ↓
    P(click)      P(conversion)
         ↓             ↓
    Loss_CTR      Loss_CVR
         └──────┬──────┘
                ↓
         Total Loss = α*Loss_CTR + β*Loss_CVR

====================================================================
📊 数据标注问题
====================================================================

关键问题：转化样本很少！

样本分布：
- 曝光：100%
- 点击：10%（10个人里1个点）
- 转化：1%（100个人里1个转化）

标注：
- CTR 任务：有点击 label（点/不点）
- CVR 任务：只有点击后才有转化 label
  - 未点击的样本：转化 label = ?（无法知道）
  - 点击的样本：转化 label = 0/1

解决方案（ESMM模型）：
- 引入 CTCVR（点击且转化的概率）
- P(CTCVR) = P(CTR) × P(CVR)
- 三个任务联合训练

====================================================================
🔑 本实现：简化版多任务学习
====================================================================

为了便于理解，我们实现 Shared-Bottom 架构：

1. 共享底层：Embedding + 共享 MLP
2. 任务专用层：CTR Tower + CVR Tower
3. 联合训练：同时优化两个损失

注意：
- CVR 任务只用点击样本训练（有转化标签的样本）
- 实际生产中会用 ESMM 等更复杂的架构

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


# ============ 1. 数据准备（带点击和转化标签）============

class MTLDataset(Dataset):
    """多任务学习数据集

    标签：
    - click: 是否点击（0/1）
    - conversion: 是否转化（0/1，只有点击后才有）
    """

    def __init__(self, num_samples=10000, num_users=1000, num_items=500):
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_items = num_items

        self.num_ages = 5
        self.num_genders = 2
        self.num_categories = 20

        # 用户画像
        self.user_ages = np.random.randint(0, self.num_ages, num_users)
        self.user_genders = np.random.randint(0, self.num_genders, num_users)

        # 物品属性
        self.item_categories = np.random.randint(0, self.num_categories, num_items)
        self.item_prices = np.random.uniform(10, 1000, num_items)

        # 生成样本
        self.samples = []
        for _ in range(num_samples):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)

            age = self.user_ages[user_id]
            gender = self.user_genders[user_id]
            category = self.item_categories[item_id]
            price = self.item_prices[item_id]

            # 模拟点击概率
            click_prob = 0.1
            if age < 2 and category < 5: click_prob += 0.3
            if gender == 1 and category in [10, 11, 12]: click_prob += 0.3

            click = 1 if np.random.rand() < click_prob else 0

            # 模拟转化概率（只有点击后才可能转化）
            conversion = 0
            if click == 1:
                cvr = 0.1  # 基础转化率10%
                # 价格影响：低价更容易转化
                if price < 200: cvr += 0.3
                elif price < 500: cvr += 0.1
                # 年轻人更冲动消费
                if age < 2: cvr += 0.2

                conversion = 1 if np.random.rand() < cvr else 0

            self.samples.append({
                'user_id': user_id,
                'item_id': item_id,
                'age': age,
                'gender': gender,
                'category': category,
                'price': price,
                'click': click,
                'conversion': conversion
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
            'click': torch.FloatTensor([sample['click']]),
            'conversion': torch.FloatTensor([sample['conversion']])
        }


# ============ 2. 多任务学习模型（Shared-Bottom）============

class MTLModel(nn.Module):
    """多任务学习模型

    架构：
    - 共享层：Embedding + 共享 MLP
    - CTR Tower：点击预测专用层
    - CVR Tower：转化预测专用层
    """

    def __init__(self, num_users, num_items, num_ages, num_genders, num_categories,
                 embedding_dim=16, shared_dim=128, tower_dim=64):
        super().__init__()

        # ============ 共享 Embedding 层 ============
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim // 2)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 4)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)

        # 计算拼接后的特征维度
        input_dim = embedding_dim * 2 + embedding_dim // 2 * 2 + embedding_dim // 4 + 1  # +1 for price

        # ============ 共享 MLP 层（底层）============
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ============ CTR Tower（点击预测）============
        self.ctr_tower = nn.Sequential(
            nn.Linear(shared_dim, tower_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(tower_dim, 1)
        )

        # ============ CVR Tower（转化预测）============
        self.cvr_tower = nn.Sequential(
            nn.Linear(shared_dim, tower_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(tower_dim, 1)
        )

    def forward(self, user_id, item_id, age, gender, category, price):
        """
        前向传播

        返回:
            ctr_logit: (batch,) - 点击预测 logit
            cvr_logit: (batch,) - 转化预测 logit
        """
        # 1. Embedding
        user_emb = self.user_embedding(user_id).squeeze(1)
        item_emb = self.item_embedding(item_id).squeeze(1)
        age_emb = self.age_embedding(age).squeeze(1)
        gender_emb = self.gender_embedding(gender).squeeze(1)
        category_emb = self.category_embedding(category).squeeze(1)

        # 2. 拼接特征
        features = torch.cat([
            user_emb, item_emb, age_emb, gender_emb, category_emb, price
        ], dim=1)

        # 3. 共享层
        shared_repr = self.shared_layers(features)

        # 4. 任务专用层
        ctr_logit = self.ctr_tower(shared_repr).squeeze(1)
        cvr_logit = self.cvr_tower(shared_repr).squeeze(1)

        return ctr_logit, cvr_logit


# ============ 3. 训练 ============

def train_model(model, train_loader, val_loader, device, num_epochs=30, lr=0.001):
    """训练多任务模型"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_ctr_loss': [], 'train_ctr_auc': [],
        'train_cvr_loss': [], 'train_cvr_auc': [],
        'val_ctr_loss': [], 'val_ctr_auc': [],
        'val_cvr_loss': [], 'val_cvr_auc': []
    }

    print("\n开始训练...")
    for epoch in range(num_epochs):
        model.train()
        train_ctr_loss = 0
        train_cvr_loss = 0
        train_ctr_preds = []
        train_ctr_labels = []
        train_cvr_preds = []
        train_cvr_labels = []

        for batch in train_loader:
            user_id = batch['user_id'].to(device)
            item_id = batch['item_id'].to(device)
            age = batch['age'].to(device)
            gender = batch['gender'].to(device)
            category = batch['category'].to(device)
            price = batch['price'].to(device)
            click = batch['click'].to(device).squeeze()
            conversion = batch['conversion'].to(device).squeeze()

            optimizer.zero_grad()

            # 前向传播
            ctr_logit, cvr_logit = model(user_id, item_id, age, gender, category, price)

            # CTR 损失（所有样本）
            ctr_loss = criterion(ctr_logit, click)

            # CVR 损失（只用点击样本）
            click_mask = (click == 1)
            if click_mask.sum() > 0:
                cvr_logit_clicked = cvr_logit[click_mask]
                conversion_clicked = conversion[click_mask]
                cvr_loss = criterion(cvr_logit_clicked, conversion_clicked)
            else:
                cvr_loss = torch.tensor(0.0).to(device)

            # 总损失（加权）
            loss = ctr_loss + cvr_loss  # 可以调整权重

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_ctr_loss += ctr_loss.item()
            train_ctr_preds.extend(torch.sigmoid(ctr_logit).detach().cpu().numpy())
            train_ctr_labels.extend(click.cpu().numpy())

            if click_mask.sum() > 0:
                train_cvr_loss += cvr_loss.item()
                train_cvr_preds.extend(torch.sigmoid(cvr_logit_clicked).detach().cpu().numpy())
                train_cvr_labels.extend(conversion_clicked.cpu().numpy())

        train_ctr_loss /= len(train_loader)
        train_ctr_auc = roc_auc_score(train_ctr_labels, train_ctr_preds)

        if len(train_cvr_labels) > 0:
            train_cvr_loss /= len(train_loader)
            train_cvr_auc = roc_auc_score(train_cvr_labels, train_cvr_preds) if len(set(train_cvr_labels)) > 1 else 0.5
        else:
            train_cvr_loss = 0
            train_cvr_auc = 0.5

        # 验证
        model.eval()
        val_ctr_loss = 0
        val_cvr_loss = 0
        val_ctr_preds = []
        val_ctr_labels = []
        val_cvr_preds = []
        val_cvr_labels = []

        with torch.no_grad():
            for batch in val_loader:
                user_id = batch['user_id'].to(device)
                item_id = batch['item_id'].to(device)
                age = batch['age'].to(device)
                gender = batch['gender'].to(device)
                category = batch['category'].to(device)
                price = batch['price'].to(device)
                click = batch['click'].to(device).squeeze()
                conversion = batch['conversion'].to(device).squeeze()

                ctr_logit, cvr_logit = model(user_id, item_id, age, gender, category, price)

                ctr_loss = criterion(ctr_logit, click)
                val_ctr_loss += ctr_loss.item()
                val_ctr_preds.extend(torch.sigmoid(ctr_logit).cpu().numpy())
                val_ctr_labels.extend(click.cpu().numpy())

                click_mask = (click == 1)
                if click_mask.sum() > 0:
                    cvr_logit_clicked = cvr_logit[click_mask]
                    conversion_clicked = conversion[click_mask]
                    cvr_loss = criterion(cvr_logit_clicked, conversion_clicked)
                    val_cvr_loss += cvr_loss.item()
                    val_cvr_preds.extend(torch.sigmoid(cvr_logit_clicked).cpu().numpy())
                    val_cvr_labels.extend(conversion_clicked.cpu().numpy())

        val_ctr_loss /= len(val_loader)
        val_ctr_auc = roc_auc_score(val_ctr_labels, val_ctr_preds)

        if len(val_cvr_labels) > 0:
            val_cvr_loss /= len(val_loader)
            val_cvr_auc = roc_auc_score(val_cvr_labels, val_cvr_preds) if len(set(val_cvr_labels)) > 1 else 0.5
        else:
            val_cvr_loss = 0
            val_cvr_auc = 0.5

        # 记录
        history['train_ctr_loss'].append(train_ctr_loss)
        history['train_ctr_auc'].append(train_ctr_auc)
        history['train_cvr_loss'].append(train_cvr_loss)
        history['train_cvr_auc'].append(train_cvr_auc)
        history['val_ctr_loss'].append(val_ctr_loss)
        history['val_ctr_auc'].append(val_ctr_auc)
        history['val_cvr_loss'].append(val_cvr_loss)
        history['val_cvr_auc'].append(val_cvr_auc)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  CTR - Train Loss: {train_ctr_loss:.4f}, Train AUC: {train_ctr_auc:.4f}')
            print(f'        Val Loss: {val_ctr_loss:.4f}, Val AUC: {val_ctr_auc:.4f}')
            print(f'  CVR - Train Loss: {train_cvr_loss:.4f}, Train AUC: {train_cvr_auc:.4f}')
            print(f'        Val Loss: {val_cvr_loss:.4f}, Val AUC: {val_cvr_auc:.4f}')

    return history


# ============ 4. 可视化 ============

def plot_training_history(history):
    """绘制训练历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # CTR Loss
    ax1.plot(history['train_ctr_loss'], label='Train CTR Loss')
    ax1.plot(history['val_ctr_loss'], label='Val CTR Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('CTR Loss')
    ax1.legend()
    ax1.grid(True)

    # CTR AUC
    ax2.plot(history['train_ctr_auc'], label='Train CTR AUC')
    ax2.plot(history['val_ctr_auc'], label='Val CTR AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('CTR AUC')
    ax2.legend()
    ax2.grid(True)

    # CVR Loss
    ax3.plot(history['train_cvr_loss'], label='Train CVR Loss')
    ax3.plot(history['val_cvr_loss'], label='Val CVR Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('CVR Loss')
    ax3.legend()
    ax3.grid(True)

    # CVR AUC
    ax4.plot(history['train_cvr_auc'], label='Train CVR AUC')
    ax4.plot(history['val_cvr_auc'], label='Val CVR AUC')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC')
    ax4.set_title('CVR AUC')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('mtl_training.png', dpi=150)
    print("\n训练历史已保存到 mtl_training.png")
    plt.close()


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  多任务学习 (Multi-Task Learning) - PyTorch实现")
    print("  同时预测点击和转化")
    print("=" * 60)

    print(f"\n使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("创建数据集（带点击和转化标签）")
    print("=" * 60)

    train_dataset = MTLDataset(num_samples=50000, num_users=1000, num_items=500)
    val_dataset = MTLDataset(num_samples=10000, num_users=1000, num_items=500)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 统计点击率和转化率
    train_clicks = [s['click'] for s in train_dataset.samples]
    train_conversions = [s['conversion'] for s in train_dataset.samples]
    ctr = sum(train_clicks) / len(train_clicks)
    cvr = sum(train_conversions) / sum(train_clicks) if sum(train_clicks) > 0 else 0

    print(f"点击率 (CTR): {ctr:.2%}")
    print(f"转化率 (CVR): {cvr:.2%}")
    print(f"点击且转化率 (CTCVR): {sum(train_conversions)/len(train_conversions):.2%}")

    print("\n" + "=" * 60)
    print("创建多任务学习模型")
    print("=" * 60)

    model = MTLModel(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        num_ages=train_dataset.num_ages,
        num_genders=train_dataset.num_genders,
        num_categories=train_dataset.num_categories,
        embedding_dim=16,
        shared_dim=128,
        tower_dim=64
    ).to(DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    history = train_model(model, train_loader, val_loader, DEVICE, num_epochs=20, lr=0.001)

    plot_training_history(history)

    print("\n" + "=" * 60)
    print("测试预测")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for i in range(5):
            sample = val_dataset[i]

            user_id = sample['user_id'].unsqueeze(0).to(DEVICE)
            item_id = sample['item_id'].unsqueeze(0).to(DEVICE)
            age = sample['age'].unsqueeze(0).to(DEVICE)
            gender = sample['gender'].unsqueeze(0).to(DEVICE)
            category = sample['category'].unsqueeze(0).to(DEVICE)
            price = sample['price'].unsqueeze(0).to(DEVICE)

            ctr_logit, cvr_logit = model(user_id, item_id, age, gender, category, price)
            ctr_prob = torch.sigmoid(ctr_logit).item()
            cvr_prob = torch.sigmoid(cvr_logit).item()

            true_click = sample['click'].item()
            true_conversion = sample['conversion'].item()

            print(f"\n样本 {i+1}:")
            print(f"  用户ID: {user_id.item()}, 物品ID: {item_id.item()}, 价格: {price.item():.1f}")
            print(f"  预测点击概率: {ctr_prob:.3f}, 真实: {int(true_click)}")
            print(f"  预测转化概率: {cvr_prob:.3f}, 真实: {int(true_conversion)}")
            print(f"  预测点击且转化: {ctr_prob * cvr_prob:.3f}")

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. 多任务学习核心
   ✓ 同时预测多个相关任务（CTR + CVR）
   ✓ 共享底层特征表示
   ✓ 任务之间互相辅助学习

2. Shared-Bottom 架构
   ✓ 共享层：所有任务共用
   ✓ Tower 层：每个任务专用
   ✓ 联合训练：同时优化多个损失

3. 为什么有效？
   ✓ 特征共享：减少参数，防止过拟合
   ✓ 任务相关性：点击和转化有关联
   ✓ 数据增强：转化样本少，借助点击样本

4. 关键技术点
   ✓ CVR 只用点击样本训练（有转化标签）
   ✓ 损失加权：平衡不同任务的重要性
   ✓ 样本不平衡：转化样本远少于点击

5. 工业应用
   ✓ 阿里 ESMM：Entire Space Multi-Task Model
   ✓ 同时优化 CTR + CVR + CTCVR
   ✓ 解决 CVR 样本选择偏差问题

6. 进阶架构
   → MMoE (Multi-gate Mixture-of-Experts)
   → PLE (Progressive Layered Extraction)
   → 任务冲突缓解

7. 业务价值
   ✓ 优化真正的业务目标（转化/GMV）
   ✓ 不只是点击量，更要成交额
   ✓ 提升推荐系统的商业价值
    """)

    print("\n✅ 多任务学习完成！")
    print("\n提示: 多任务学习是推荐系统提升业务价值的关键技术")


if __name__ == "__main__":
    main()
