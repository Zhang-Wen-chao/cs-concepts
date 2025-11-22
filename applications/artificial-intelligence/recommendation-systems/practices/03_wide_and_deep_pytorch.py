"""
Wide & Deep 模型 - PyTorch实现
推荐系统精排阶段的经典架构（Google 2016）

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 推荐系统链路中的位置
====================================================================

用户请求
   ↓
1. 召回（Retrieval）✅ 已学习
   - 双塔模型：百万 → 几千
   ↓
2. 粗排（Pre-Ranking，可选）
   - 几千 → 几百
   ↓
3. 精排（Ranking）← 我们在这里！
   - Wide & Deep：几百 → 几十
   - 精准预测用户点击/购买概率
   ↓
4. 重排（Re-Ranking）
   - 多样性、打散
   ↓
展示给用户

====================================================================
🎯 为什么需要 Wide & Deep？
====================================================================

召回阶段的问题：
- 双塔模型：只能用点积计算相似度
- 无法建模特征之间的交叉（如：年龄×性别、类别×价格）
- 只适合快速筛选，不适合精准排序

精排阶段的需求：
✓ 精准预测点击率/转化率（CTR/CVR）
✓ 利用丰富的特征交叉
✓ 平衡记忆能力和泛化能力

====================================================================
🏗️ Wide & Deep 架构
====================================================================

                   用户特征 + 物品特征 + 交互特征
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
    Wide 部分                               Deep 部分
  (线性模型)                              (深度网络)
        ↓                                       ↓
  交叉特征 ────────────┐              ┌──── Embedding + MLP
  [age×gender,         │              │     [user_emb,
   category×price]     │              │      item_emb,
        ↓              │              │      age, gender...]
   Linear(稀疏)        │              │          ↓
        ↓              │              │      MLP(256→128→64)
     logit_wide ───────┴──────────────┴──── logit_deep
                            ↓
                      logit = logit_wide + logit_deep
                            ↓
                      sigmoid(logit) → 预测概率 (0-1)

====================================================================
🔑 Wide vs Deep 的区别
====================================================================

Wide 部分（记忆能力 - Memorization）：
- 线性模型：y = w₁x₁ + w₂x₂ + ... + b
- 特征：人工设计的交叉特征（如 AND(gender=female, category=美妆)）
- 优点：记住训练数据中的规则（女性 + 美妆 → 高点击）
- 缺点：无法泛化到未见过的组合

Deep 部分（泛化能力 - Generalization）：
- 深度网络：自动学习特征表示
- 特征：原始特征 + Embedding
- 优点：泛化到新组合（类似美妆的类别也可能被推荐）
- 缺点：可能忽略重要的规则

组合的好处：
✓ Wide：记住确定的规则（如促销商品 + 价格敏感用户）
✓ Deep：发现潜在的模式（如相似用户的行为）
✓ 互补：既精准又能泛化

====================================================================
📊 数据格式示例
====================================================================

输入特征：
{
    # 用户特征
    'user_id': 123,
    'age': 25,
    'gender': 1,
    'city': 'Beijing',

    # 物品特征
    'item_id': 456,
    'category': 'Electronics',
    'price': 999,
    'brand': 'Apple',

    # 交互特征
    'hour': 14,  # 访问时间
    'device': 'mobile'
}

输出：
- label: 1 (点击) / 0 (未点击)
- 预测: 0.85 (85% 概率会点击)

====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ 1. 数据准备 ============

class CTRDataset(Dataset):
    """点击率预测数据集（模拟电商场景）

    特征：
    - 用户：user_id, age, gender, city
    - 物品：item_id, category, price, brand
    - 上下文：hour, device

    标签：是否点击 (0/1)
    """

    def __init__(self, num_samples=10000, num_users=1000, num_items=500):
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_items = num_items

        # 特征空间大小
        self.num_ages = 5       # 5个年龄段
        self.num_genders = 2    # 2种性别
        self.num_cities = 10    # 10个城市
        self.num_categories = 20  # 20个类目
        self.num_brands = 50    # 50个品牌
        self.num_hours = 24     # 24小时
        self.num_devices = 2    # 2种设备

        # 生成用户画像
        self.user_ages = np.random.randint(0, self.num_ages, num_users)
        self.user_genders = np.random.randint(0, self.num_genders, num_users)
        self.user_cities = np.random.randint(0, self.num_cities, num_users)

        # 生成物品属性
        self.item_categories = np.random.randint(0, self.num_categories, num_items)
        self.item_brands = np.random.randint(0, self.num_brands, num_items)
        self.item_prices = np.random.uniform(10, 1000, num_items)  # 价格10-1000

        # 生成交互数据
        self.samples = []
        for _ in range(num_samples):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)
            hour = np.random.randint(0, self.num_hours)
            device = np.random.randint(0, self.num_devices)

            # 模拟点击规律（简化版）
            age = self.user_ages[user_id]
            gender = self.user_genders[user_id]
            category = self.item_categories[item_id]
            price = self.item_prices[item_id]

            # 规则1：年轻人(age<2) + 电子产品(category<5) → 高点击
            rule1 = (age < 2 and category < 5)

            # 规则2：女性(gender=1) + 美妆(category in [10,11,12]) → 高点击
            rule2 = (gender == 1 and category in [10, 11, 12])

            # 规则3：晚上(hour>18) + 娱乐(category in [15,16,17]) → 高点击
            rule3 = (hour > 18 and category in [15, 16, 17])

            # 规则4：低价(<100) → 高点击
            rule4 = (price < 100)

            # 综合判断
            click_prob = 0.1  # 基础点击率
            if rule1: click_prob += 0.4
            if rule2: click_prob += 0.4
            if rule3: click_prob += 0.3
            if rule4: click_prob += 0.2

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
            # ID特征
            'user_id': torch.LongTensor([sample['user_id']]),
            'item_id': torch.LongTensor([sample['item_id']]),

            # 类别特征
            'age': torch.LongTensor([sample['age']]),
            'gender': torch.LongTensor([sample['gender']]),
            'city': torch.LongTensor([sample['city']]),
            'category': torch.LongTensor([sample['category']]),
            'brand': torch.LongTensor([sample['brand']]),
            'hour': torch.LongTensor([sample['hour']]),
            'device': torch.LongTensor([sample['device']]),

            # 数值特征
            'price': torch.FloatTensor([sample['price']]),

            # 标签
            'label': torch.FloatTensor([sample['label']])
        }


# ============ 2. Wide & Deep 模型 ============

class WideAndDeepModel(nn.Module):
    """Wide & Deep 模型

    Wide 部分：线性模型 + 交叉特征
    Deep 部分：Embedding + MLP
    """

    def __init__(self,
                 num_users, num_items, num_ages, num_genders, num_cities,
                 num_categories, num_brands, num_hours, num_devices,
                 embedding_dim=16, hidden_dims=[256, 128, 64]):
        super().__init__()

        # ============ Deep 部分 ============

        # Embedding 层（将ID和类别特征映射为稠密向量）
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.age_embedding = nn.Embedding(num_ages, embedding_dim // 2)
        self.gender_embedding = nn.Embedding(num_genders, embedding_dim // 4)
        self.city_embedding = nn.Embedding(num_cities, embedding_dim // 2)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim // 2)
        self.brand_embedding = nn.Embedding(num_brands, embedding_dim // 2)
        self.hour_embedding = nn.Embedding(num_hours, embedding_dim // 4)
        self.device_embedding = nn.Embedding(num_devices, embedding_dim // 4)

        # 计算 Deep 部分的输入维度
        deep_input_dim = (
            embedding_dim * 2 +              # user + item
            (embedding_dim // 2) * 4 +       # age + city + category + brand
            (embedding_dim // 4) * 3 +       # gender + hour + device
            1                                # price (数值特征)
        )

        # Deep MLP
        layers = []
        input_dim = deep_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.deep_mlp = nn.Sequential(*layers)
        self.deep_output = nn.Linear(hidden_dims[-1], 1)

        # ============ Wide 部分 ============

        # Wide 部分使用原始特征 + 交叉特征
        # 这里简化处理：为每种交叉组合创建 Embedding
        # 实际生产中会用稀疏矩阵或 Feature Hashing

        # 交叉特征1：age × gender（5 × 2 = 10种组合）
        self.cross_age_gender = nn.Embedding(num_ages * num_genders, 1)

        # 交叉特征2：category × gender（20 × 2 = 40种组合）
        self.cross_category_gender = nn.Embedding(num_categories * num_genders, 1)

        # 交叉特征3：hour × device（24 × 2 = 48种组合）
        self.cross_hour_device = nn.Embedding(num_hours * num_devices, 1)

        # Wide 的线性输出
        self.wide_output = nn.Linear(3, 1)  # 3个交叉特征

        # 保存特征空间大小（用于计算交叉特征ID）
        self.num_genders = num_genders
        self.num_devices = num_devices

    def forward(self, user_id, item_id, age, gender, city, category, brand,
                price, hour, device):
        """
        前向传播

        返回:
            logit: (batch_size,) - 预测的 logit（未经过sigmoid）
        """
        batch_size = user_id.size(0)

        # ============ Deep 部分 ============

        # 1. Embedding
        user_emb = self.user_embedding(user_id).squeeze(1)      # (batch, emb_dim)
        item_emb = self.item_embedding(item_id).squeeze(1)
        age_emb = self.age_embedding(age).squeeze(1)
        gender_emb = self.gender_embedding(gender).squeeze(1)
        city_emb = self.city_embedding(city).squeeze(1)
        category_emb = self.category_embedding(category).squeeze(1)
        brand_emb = self.brand_embedding(brand).squeeze(1)
        hour_emb = self.hour_embedding(hour).squeeze(1)
        device_emb = self.device_embedding(device).squeeze(1)

        # 2. 拼接所有特征
        deep_input = torch.cat([
            user_emb, item_emb, age_emb, gender_emb, city_emb,
            category_emb, brand_emb, hour_emb, device_emb, price
        ], dim=1)

        # 3. 通过 MLP
        deep_hidden = self.deep_mlp(deep_input)
        logit_deep = self.deep_output(deep_hidden).squeeze(1)  # (batch,)

        # ============ Wide 部分 ============

        # 1. 构造交叉特征ID
        # age × gender: 将(age, gender)映射为一个ID
        cross_id_1 = (age * self.num_genders + gender).squeeze(1)  # (batch,)

        # category × gender
        cross_id_2 = (category * self.num_genders + gender).squeeze(1)

        # hour × device
        cross_id_3 = (hour * self.num_devices + device).squeeze(1)

        # 2. 获取交叉特征的权重
        cross_feat_1 = self.cross_age_gender(cross_id_1).squeeze(1)  # (batch,)
        cross_feat_2 = self.cross_category_gender(cross_id_2).squeeze(1)
        cross_feat_3 = self.cross_hour_device(cross_id_3).squeeze(1)

        # 3. Wide 的线性组合
        wide_input = torch.stack([cross_feat_1, cross_feat_2, cross_feat_3], dim=1)  # (batch, 3)
        logit_wide = self.wide_output(wide_input).squeeze(1)  # (batch,)

        # ============ 组合 Wide + Deep ============

        logit = logit_wide + logit_deep

        return logit


# ============ 3. 训练 ============

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.001):
    """训练 Wide & Deep 模型"""
    # 二分类使用 BCEWithLogitsLoss（内置sigmoid，数值更稳定）
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            # 提取特征
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

            # 前向传播
            logit = model(user_id, item_id, age, gender, city, category,
                         brand, price, hour, device_feat)

            # 计算损失
            loss = criterion(logit, label)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logit).detach().cpu().numpy())
            train_labels.extend(label.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)

        # 验证阶段
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
    plt.savefig('wide_and_deep_training.png', dpi=150)
    print("\n训练历史已保存到 wide_and_deep_training.png")
    plt.close()


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  Wide & Deep 模型 - PyTorch实现")
    print("  推荐系统精排阶段的经典架构")
    print("=" * 60)

    # 检查设备
    print(f"\n使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # 创建数据集
    print("\n" + "=" * 60)
    print("创建点击率预测数据集")
    print("=" * 60)

    train_dataset = CTRDataset(num_samples=20000, num_users=1000, num_items=500)
    val_dataset = CTRDataset(num_samples=5000, num_users=1000, num_items=500)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 计算正负样本比例
    train_labels = [s['label'] for s in train_dataset.samples]
    pos_ratio = sum(train_labels) / len(train_labels)
    print(f"正样本比例: {pos_ratio:.2%}")

    # 创建模型
    print("\n" + "=" * 60)
    print("创建 Wide & Deep 模型")
    print("=" * 60)

    model = WideAndDeepModel(
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

    # 训练模型
    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    history = train_model(model, train_loader, val_loader, DEVICE, num_epochs=30, lr=0.001)

    # 可视化
    plot_training_history(history)

    # 测试预测
    print("\n" + "=" * 60)
    print("测试预测")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for i in range(3):
            sample = val_dataset[i]

            # 准备输入
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

    # 总结
    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. Wide & Deep 架构
   ✓ Wide 部分: 线性模型 + 交叉特征 → 记忆能力
   ✓ Deep 部分: Embedding + MLP → 泛化能力
   ✓ 组合: logit = logit_wide + logit_deep

2. Wide 部分（记忆）
   ✓ 交叉特征: age×gender, category×gender, hour×device
   ✓ 捕获确定的规则: "女性+美妆→高点击"
   ✓ 实现方式: Embedding（简化版）或稀疏矩阵

3. Deep 部分（泛化）
   ✓ Embedding: 将ID/类别映射为稠密向量
   ✓ MLP: 自动学习特征组合
   ✓ 泛化到新组合: 类似的用户/物品

4. 与双塔模型的区别
   双塔模型（召回）:
   - 用户塔 + 物品塔独立
   - 只能用点积计算相似度
   - 快速但不精准

   Wide & Deep（精排）:
   - 用户和物品特征一起输入
   - 可以建模任意特征交叉
   - 慢但精准

5. 评价指标
   ✓ AUC (Area Under Curve): 排序能力
   ✓ LogLoss: 概率校准程度
   ✓ 精排关注预测的准确性，不只是排序

6. 工业实践
   ✓ Wide特征: 需要人工设计（领域知识）
   ✓ 特征工程: 统计特征、序列特征、交叉特征
   ✓ 在线服务: Wide和Deep都在线推理（vs 双塔的离线索引）

7. 下一步
   → DeepFM: 自动学习交叉特征（无需人工设计）
   → DIN: 用户兴趣建模（注意力机制）
   → 多任务学习: 同时预测点击+转化
    """)

    print("\n✅ Wide & Deep 学习完成！")
    print("\n提示: Wide & Deep 是精排的基础，接下来学习 DeepFM 可以自动化特征交叉")


if __name__ == "__main__":
    main()
