"""
DIN (Deep Interest Network) - PyTorch实现
基于 Attention 的用户兴趣建模（阿里巴巴 2018）

作者: Zhang Wenchao
日期: 2025-11-22

====================================================================
📖 从 DeepFM 到 DIN
====================================================================

DeepFM 的问题：
- 用户历史行为 → 固定的 Embedding
- 所有候选物品都用同一个用户表示
- 无法体现用户兴趣的多样性

例子：
用户历史：[手机, 电脑, 小说, 耳机, 图书]
候选1：笔记本电脑 → 应该关注 [手机, 电脑, 耳机]
候选2：推理小说   → 应该关注 [小说, 图书]

但 DeepFM 对两个候选都用同一个用户向量！

====================================================================
🎯 DIN 的核心思想
====================================================================

Attention 机制：根据候选物品，动态计算用户历史的权重

用户表示 = Σ attention_weight_i × history_item_i
            ↑ 根据候选物品动态计算

公式：
attention_weight_i = softmax(f(candidate_item, history_item_i))

其中 f 是一个小网络（MLP），学习候选和历史的相关性。

====================================================================
🏗️ DIN 架构
====================================================================

输入：
- 用户特征：[user_id, age, gender]
- 候选物品：[item_id, category]
- 用户历史：[历史item_1, 历史item_2, ..., 历史item_n]

流程：

1️⃣ Embedding
   用户特征 → user_emb
   候选物品 → candidate_emb
   历史物品 → [hist_emb_1, hist_emb_2, ..., hist_emb_n]

2️⃣ Attention Layer（核心！）
   for each hist_emb_i:
       # 计算候选物品和历史物品的相关性
       score_i = MLP([candidate_emb, hist_emb_i, candidate_emb - hist_emb_i, candidate_emb * hist_emb_i])

   # Softmax 归一化
   attention_weights = softmax([score_1, score_2, ..., score_n])

   # 加权求和
   user_interest = Σ attention_weight_i × hist_emb_i

3️⃣ Concatenate
   final_input = [user_emb, candidate_emb, user_interest]

4️⃣ MLP
   logit = MLP(final_input)

5️⃣ Sigmoid
   click_prob = sigmoid(logit)

====================================================================
🔑 Attention 计算细节
====================================================================

输入：
- candidate_emb: (batch, emb_dim) - 候选物品
- hist_embs: (batch, seq_len, emb_dim) - 历史物品序列

计算相关性特征（4种）：
1. candidate_emb (复制seq_len次)
2. hist_emb_i
3. candidate_emb - hist_emb_i  (差值，衡量距离)
4. candidate_emb * hist_emb_i  (逐元素乘，衡量相似度)

拼接: [candidate, history, sub, mul] → (batch, seq_len, 4*emb_dim)

通过 Attention MLP:
score_i = MLP(concat_features_i)  → (batch, seq_len, 1)

Softmax:
attention_weights = softmax(scores)  → (batch, seq_len, 1)

加权求和:
user_interest = Σ attention_weight_i × hist_emb_i  → (batch, emb_dim)

====================================================================
💡 DIN vs 之前的模型
====================================================================

| 模型 | 用户表示 | 优缺点 |
|------|---------|--------|
| 双塔 | 固定向量 | ❌ 无法体现兴趣多样性 |
| Wide & Deep | 固定向量 | ❌ 历史行为平均池化 |
| DeepFM | 固定向量 | ❌ 所有候选用同一表示 |
| **DIN** | **动态向量** | ✅ 根据候选动态关注历史<br>✅ 体现兴趣多样性 |

====================================================================
🧮 Attention 的好处
====================================================================

1. 可解释性：
   - 可以看到模型关注了哪些历史行为
   - 例如：推荐"笔记本"时，关注了"手机(0.5)"、"电脑(0.4)"、"耳机(0.1)"

2. 效果提升：
   - 阿里巴巴论文：CTR 提升 ~10%
   - 特别适合用户兴趣多样的场景

3. 灵活性：
   - 历史序列长度可变
   - 自动学习哪些历史重要

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


# ============ 1. 数据准备（加入用户历史序列）============

class DINDataset(Dataset):
    """带用户历史序列的 CTR 数据集

    新增：用户历史物品序列
    """

    def __init__(self, num_samples=10000, num_users=1000, num_items=500, max_hist_len=10):
        self.num_samples = num_samples
        self.num_users = num_users
        self.num_items = num_items
        self.max_hist_len = max_hist_len

        self.num_categories = 20

        # 物品属性
        self.item_categories = np.random.randint(0, self.num_categories, num_items)

        # 生成用户历史（每个用户有一个历史物品列表）
        self.user_histories = {}
        for user_id in range(num_users):
            # 每个用户的历史长度随机（1到max_hist_len）
            hist_len = np.random.randint(1, max_hist_len + 1)
            history = np.random.randint(0, num_items, hist_len)
            self.user_histories[user_id] = history

        # 生成训练样本
        self.samples = []
        for _ in range(num_samples):
            user_id = np.random.randint(0, num_users)
            candidate_item = np.random.randint(0, num_items)

            # 获取用户历史
            history = self.user_histories[user_id]

            # 模拟点击规律：更复杂的规则，让模型有东西可学
            candidate_cat = self.item_categories[candidate_item]
            history_cats = self.item_categories[history]

            # 规则1：类别完全匹配的物品数量
            exact_match = np.sum(history_cats == candidate_cat)

            # 规则2：类别接近的物品数量（相差1或2的类别也算相关）
            close_match = np.sum((np.abs(history_cats - candidate_cat) <= 2))

            # 规则3：最近的历史物品权重更高
            if len(history) >= 3:
                recent_match = 1.0 if candidate_cat in history_cats[-3:] else 0.0
            else:
                recent_match = 1.0 if candidate_cat in history_cats else 0.0

            # 规则4：候选物品ID也影响点击（某些物品本身就热门）
            item_popularity = 1.0 if candidate_item % 10 == 0 else 0.0

            # 综合计算点击概率
            click_prob = 0.05  # 基础概率
            click_prob += (exact_match / len(history)) * 0.4  # 完全匹配贡献40%
            click_prob += (close_match / len(history)) * 0.2   # 接近匹配贡献20%
            click_prob += recent_match * 0.2                   # 最近历史贡献20%
            click_prob += item_popularity * 0.15               # 物品热度贡献15%

            label = 1 if np.random.rand() < click_prob else 0

            self.samples.append({
                'user_id': user_id,
                'candidate_item': candidate_item,
                'history': history,
                'label': label
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        history = sample['history']

        # Padding：历史序列不足max_hist_len时用0填充
        hist_len = len(history)
        padded_history = np.zeros(self.max_hist_len, dtype=np.int64)
        padded_history[:hist_len] = history

        # Mask：标记哪些是真实历史（1），哪些是padding（0）
        hist_mask = np.zeros(self.max_hist_len, dtype=np.float32)
        hist_mask[:hist_len] = 1.0

        return {
            'user_id': torch.LongTensor([sample['user_id']]),
            'candidate_item': torch.LongTensor([sample['candidate_item']]),
            'history': torch.LongTensor(padded_history),  # (max_hist_len,)
            'hist_mask': torch.FloatTensor(hist_mask),    # (max_hist_len,)
            'label': torch.FloatTensor([sample['label']])
        }


# ============ 2. Attention Layer ============

class AttentionLayer(nn.Module):
    """DIN 的 Attention 层

    根据候选物品，动态计算用户历史的权重
    """

    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()

        # Attention MLP
        # 输入：[candidate, history, candidate-history, candidate*history]
        # 维度：4 * embedding_dim
        self.attention_mlp = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出每个历史物品的分数
        )

    def forward(self, candidate_emb, hist_embs, hist_mask):
        """
        参数:
            candidate_emb: (batch, emb_dim) - 候选物品 Embedding
            hist_embs: (batch, seq_len, emb_dim) - 历史物品 Embedding 序列
            hist_mask: (batch, seq_len) - 历史序列的 mask（0=padding, 1=真实）

        返回:
            user_interest: (batch, emb_dim) - 用户兴趣向量
            attention_weights: (batch, seq_len) - 注意力权重（用于可视化）
        """
        batch_size, seq_len, emb_dim = hist_embs.size()

        # 1. 将 candidate_emb 扩展到和 hist_embs 相同的维度
        candidate_emb_expand = candidate_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, emb_dim)

        # 2. 构造 4 种相关性特征
        sub = candidate_emb_expand - hist_embs  # 差值
        mul = candidate_emb_expand * hist_embs  # 逐元素乘

        # 3. 拼接特征
        concat_features = torch.cat([
            candidate_emb_expand,  # 候选
            hist_embs,             # 历史
            sub,                   # 差值
            mul                    # 乘积
        ], dim=2)  # (batch, seq_len, 4*emb_dim)

        # 4. 通过 Attention MLP 计算分数
        scores = self.attention_mlp(concat_features).squeeze(2)  # (batch, seq_len)

        # 5. Mask：将 padding 位置的分数设为很小的值（-inf），softmax 后权重为0
        scores = scores.masked_fill(hist_mask == 0, -1e9)

        # 6. Softmax 归一化
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # 7. 加权求和得到用户兴趣向量
        user_interest = torch.sum(
            attention_weights.unsqueeze(2) * hist_embs,  # (batch, seq_len, 1) * (batch, seq_len, emb_dim)
            dim=1  # (batch, emb_dim)
        )

        return user_interest, attention_weights


# ============ 3. DIN 模型 ============

class DINModel(nn.Module):
    """DIN (Deep Interest Network) 完整模型"""

    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dims=[256, 128, 64]):
        super().__init__()

        # ============ Embedding 层 ============
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # ============ Attention 层 ============
        self.attention = AttentionLayer(embedding_dim, hidden_dim=64)

        # ============ MLP ============
        # 输入：user_emb + candidate_emb + user_interest
        mlp_input_dim = embedding_dim * 3

        layers = []
        input_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, user_id, candidate_item, history, hist_mask):
        """
        参数:
            user_id: (batch, 1) - 用户ID
            candidate_item: (batch, 1) - 候选物品ID
            history: (batch, seq_len) - 历史物品ID序列
            hist_mask: (batch, seq_len) - 历史序列mask

        返回:
            logit: (batch,) - 预测的 logit
            attention_weights: (batch, seq_len) - 注意力权重
        """
        # 1. Embedding
        user_emb = self.user_embedding(user_id).squeeze(1)  # (batch, emb_dim)
        candidate_emb = self.item_embedding(candidate_item).squeeze(1)  # (batch, emb_dim)
        hist_embs = self.item_embedding(history)  # (batch, seq_len, emb_dim)

        # 2. Attention：根据候选物品动态计算用户兴趣
        user_interest, attention_weights = self.attention(candidate_emb, hist_embs, hist_mask)

        # 3. 拼接所有特征
        final_input = torch.cat([user_emb, candidate_emb, user_interest], dim=1)

        # 4. MLP
        hidden = self.mlp(final_input)
        logit = self.output(hidden).squeeze(1)

        return logit, attention_weights


# ============ 4. 训练 ============

def train_model(model, train_loader, val_loader, device, num_epochs=30, lr=0.001):
    """训练 DIN 模型"""
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
            candidate_item = batch['candidate_item'].to(device)
            hist_items = batch['history'].to(device)
            hist_mask = batch['hist_mask'].to(device)
            label = batch['label'].to(device).squeeze()

            optimizer.zero_grad()

            logit, _ = model(user_id, candidate_item, hist_items, hist_mask)
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
                candidate_item = batch['candidate_item'].to(device)
                hist_items = batch['history'].to(device)
                hist_mask = batch['hist_mask'].to(device)
                label = batch['label'].to(device).squeeze()

                logit, _ = model(user_id, candidate_item, hist_items, hist_mask)
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


# ============ 5. 可视化 Attention ============

def visualize_attention(model, dataset, device, sample_idx=0):
    """可视化 Attention 权重"""
    model.eval()

    sample = dataset[sample_idx]
    user_id = sample['user_id'].unsqueeze(0).to(device)
    candidate_item = sample['candidate_item'].unsqueeze(0).to(device)
    hist_items = sample['history'].unsqueeze(0).to(device)
    hist_mask = sample['hist_mask'].unsqueeze(0).to(device)

    with torch.no_grad():
        logit, attention_weights = model(user_id, candidate_item, hist_items, hist_mask)

    # 获取真实历史长度
    hist_len = int(hist_mask.sum().item())
    history_items = hist_items.squeeze().cpu().numpy()[:hist_len]
    attention_weights = attention_weights.squeeze().cpu().numpy()[:hist_len]

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(hist_len), attention_weights)
    ax.set_xlabel('History Item Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Attention Weights (Candidate Item: {candidate_item.item()})')
    ax.set_xticks(range(hist_len))
    ax.set_xticklabels([f'Item {item}' for item in history_items], rotation=45)

    # 标注权重值
    for i, (bar, weight) in enumerate(zip(bars, attention_weights)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('din_attention_visualization.png', dpi=150)
    print("\nAttention 可视化已保存到 din_attention_visualization.png")
    plt.close()


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
    plt.savefig('din_training.png', dpi=150)
    print("\n训练历史已保存到 din_training.png")
    plt.close()


# ============ 主函数 ============

def main():
    print("\n" + "🚀 " + "=" * 58)
    print("  DIN (Deep Interest Network) - PyTorch实现")
    print("  基于 Attention 的用户兴趣建模")
    print("=" * 60)

    print(f"\n使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print("创建数据集（带用户历史序列）")
    print("=" * 60)

    train_dataset = DINDataset(num_samples=50000, num_users=1000, num_items=500, max_hist_len=10)  # 增加数据
    val_dataset = DINDataset(num_samples=10000, num_users=1000, num_items=500, max_hist_len=10)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"最大历史长度: {train_dataset.max_hist_len}")

    train_labels = [s['label'] for s in train_dataset.samples]
    pos_ratio = sum(train_labels) / len(train_labels)
    print(f"正样本比例: {pos_ratio:.2%}")

    print("\n" + "=" * 60)
    print("创建 DIN 模型")
    print("=" * 60)

    model = DINModel(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        embedding_dim=16,  # 降低 32 → 16
        hidden_dims=[128, 64]  # 减少层数和维度
    ).to(DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)

    history = train_model(model, train_loader, val_loader, DEVICE, num_epochs=15, lr=0.001)  # 减少到15轮

    plot_training_history(history)

    print("\n" + "=" * 60)
    print("测试预测 + Attention 可视化")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for i in range(3):
            sample = val_dataset[i]

            user_id = sample['user_id'].unsqueeze(0).to(DEVICE)
            candidate_item = sample['candidate_item'].unsqueeze(0).to(DEVICE)
            hist_items = sample['history'].unsqueeze(0).to(DEVICE)
            hist_mask = sample['hist_mask'].unsqueeze(0).to(DEVICE)

            logit, attention_weights = model(user_id, candidate_item, hist_items, hist_mask)
            pred_prob = torch.sigmoid(logit).item()
            true_label = sample['label'].item()

            hist_len = int(hist_mask.sum().item())
            history_items = hist_items.squeeze().cpu().numpy()[:hist_len]
            attn_weights = attention_weights.squeeze().cpu().numpy()[:hist_len]

            print(f"\n样本 {i+1}:")
            print(f"  用户ID: {user_id.item()}, 候选物品: {candidate_item.item()}")
            print(f"  历史物品: {history_items}")
            print(f"  Attention权重: {[f'{w:.3f}' for w in attn_weights]}")
            print(f"  预测概率: {pred_prob:.3f}")
            print(f"  真实标签: {int(true_label)}")
            print(f"  预测结果: {'点击 ✓' if pred_prob > 0.5 else '不点击 ✗'}")

    # 可视化一个样本的 Attention
    visualize_attention(model, val_dataset, DEVICE, sample_idx=0)

    print("\n" + "=" * 60)
    print("学习总结")
    print("=" * 60)

    print("""
1. DIN 核心创新
   ✓ Attention 机制：根据候选物品动态计算用户兴趣
   ✓ 用户表示不再固定，而是针对不同候选有不同表示
   ✓ 体现了用户兴趣的多样性

2. Attention 计算过程
   ✓ 相关性特征：[candidate, history, sub, mul]
   ✓ Attention MLP：计算每个历史物品的分数
   ✓ Softmax：归一化为权重
   ✓ 加权求和：得到用户兴趣向量

3. 与之前模型的对比
   DeepFM:
   - 用户表示固定
   - 所有候选物品用同一个用户向量

   DIN:
   - 用户表示动态（根据候选计算）
   - 不同候选激活不同的历史行为

4. 可解释性
   ✓ 可以看到模型关注了哪些历史行为
   ✓ 有助于理解推荐结果
   ✓ 便于调试和优化

5. 工业应用
   ✓ 阿里巴巴：淘宝、天猫推荐
   ✓ 适合用户兴趣多样的场景
   ✓ CTR 提升 ~10%（论文数据）

6. 技术要点
   ✓ Padding + Mask：处理变长序列
   ✓ Attention 权重可视化：增强可解释性
   ✓ 4种相关性特征：充分建模候选和历史的关系

7. 下一步
   → DIEN：用户兴趣进化网络（GRU + Attention）
   → 多任务学习：同时预测点击和转化
   → 序列推荐：考虑用户行为的时间顺序
    """)

    print("\n✅ DIN 学习完成！")
    print("\n提示: DIN 是阿里巴巴推荐系统的核心模型之一")
    print("      Attention 机制让模型更加灵活和可解释")


if __name__ == "__main__":
    main()
