# 推荐系统实践

> 按照真实推荐链路组织的完整实现

## 📖 完整推荐链路

```
召回（百万→几千） → 粗排（几千→几百） → 精排（几百→几十） → 重排 → 混排 → 展示
```

---

## 🗂️ 目录结构

```
practices/
├── 01_collaborative_filtering.py          # 基础：协同过滤
├── 02_recall/                             # 召回阶段
│   └── two_tower_pytorch.py              # 双塔模型：向量检索
├── 03_coarse_ranking/                     # 粗排阶段
│   └── lightweight_models_pytorch.py     # 轻量级模型：快速过滤
├── 04_ranking/                            # 精排阶段
│   ├── wide_and_deep_pytorch.py          # Wide & Deep：手工特征交叉
│   ├── deepfm_pytorch.py                 # DeepFM：自动特征交叉
│   ├── din_pytorch.py                    # DIN：用户兴趣建模（Attention）
│   └── multitask_learning_pytorch.py     # 多任务学习：CTR + CVR
├── 05_reranking/                          # 重排阶段
│   └── mmr_diversity_pytorch.py          # MMR：多样性优化
└── 06_blending/                           # 混排阶段
    └── ad_insertion_system.py            # 广告穿插与质量控制
```

---

## 📚 各阶段详解

### 01 - 协同过滤 (Collaborative Filtering)
**位置：** 传统推荐算法基础

**实现内容：**
- 用户-物品评分矩阵
- 基于用户的协同过滤 (User-based CF)
- 基于物品的协同过滤 (Item-based CF)
- 矩阵分解 (Matrix Factorization) = Embedding 的前身

**关键概念：** 矩阵分解 = 权重学习 = Embedding

---

### 02 - 召回 (Recall)
**目标：** 百万级候选 → 几千个

**核心模型：** 双塔模型（Two-Tower Model）

**实现内容：**
- 用户塔 + 物品塔
- Embedding 层
- 向量相似度计算（点积/余弦）
- 训练和推理分离
- **离线预计算**：物品向量提前算好，在线只计算用户向量

**关键优化：** ANN 向量检索（FAISS/Milvus）

---

### 03 - 粗排 (Coarse Ranking)
**目标：** 几千个候选 → 几百个

**核心思想：** 用轻量级模型快速过滤

**实现内容：**
- 双塔增强版（更多特征，更少参数）
- 轻量级 MLP（2-3层）
- 知识蒸馏（从精排模型学习）

**关键指标：**
- 参数量 < 精排的 20%
- AUC 接近精排（差距 < 5%）
- QPS > 精排的 5 倍

---

### 04 - 精排 (Fine Ranking)
**目标：** 几百个候选 → 几十个

**核心思想：** 精准预测点击/转化概率

**多种模型：**

#### 4.1 Wide & Deep
- **Wide 部分**：手工交叉特征
- **Deep 部分**：自动学习特征
- **适用场景**：有明确的业务规则

#### 4.2 DeepFM
- **FM 层**：自动学习所有二阶特征交叉
- **Deep 层**：学习高阶特征
- **核心优势**：O(nk) 复杂度，无需手工设计

#### 4.3 DIN (Deep Interest Network)
- **核心创新**：Attention 机制
- **动态用户表示**：根据候选物品关注不同的历史
- **适用场景**：用户兴趣多样的场景

#### 4.4 多任务学习 (Multi-Task Learning)
- **同时优化**：CTR（点击） + CVR（转化）
- **Shared-Bottom**：共享底层特征
- **业务价值**：优化真正的业务目标（GMV）

---

### 05 - 重排 (Re-Ranking)
**目标：** 优化列表级目标（多样性、新颖性）

**核心方法：**
- **MMR**：最大边际相关性（平衡相关性和多样性）
- **DPP**：行列式点过程（理论最优）
- **规则打散**：类别打散、价格打散、作者打散

**为什么需要：**
- 精排只考虑单个物品的点击率
- 忽略了列表级的用户体验
- 可能导致推荐结果同质化

---

### 06 - 混排 (Blending)
**目标：** 生成用户最终看到的列表

**核心任务：**
- 广告穿插（根据用户等级调整位置）
- 运营位插入（热门活动、新人专区）
- 低质过滤（违规内容、标题党）
- 频控（防止重复曝光）

**三方平衡：**
- 用户体验：广告比例 < 20%
- 平台收益：eCPM × 曝光量
- 内容质量：平均分数 > 阈值

---

## 🎯 学习路径

### 已学习 ✅
1. 协同过滤 - 理解矩阵分解
2. 双塔模型（召回） - 理解向量检索
3. Wide & Deep（精排） - 理解特征工程
4. DeepFM（精排） - 理解自动特征交叉
5. DIN（精排） - 理解 Attention 机制
6. 多任务学习 - 理解 CTR + CVR

### 待学习 🔄
7. 粗排 - 理解性能和效果的权衡
8. 重排 - 理解多样性优化
9. 混排 - 理解业务策略

---

## 💡 学习建议

### 1. 按顺序学习
- 不要跳过，每个阶段都有存在的意义
- 理解为什么需要这么多阶段（性能 vs 效果）

### 2. 关注关键指标
- **召回**：召回率、ANN 速度
- **粗排**：参数量、QPS、AUC
- **精排**：AUC、GAUC、准确率
- **重排**：多样性、覆盖度、NDCG
- **混排**：广告比例、收益、用户满意度

### 3. 理解权衡
- 每个阶段都在做权衡：性能 vs 效果
- 召回：快速但粗糙
- 粗排：快速过滤
- 精排：精准但慢
- 重排：优化整体体验
- 混排：平衡多方利益

### 4. 对比真实系统
- 你们项目的推荐系统用了哪些模块？
- 哪些阶段可以优化？
- 性能瓶颈在哪里？

---

## 🔧 运行代码

```bash
# 1. 基础算法
python 01_collaborative_filtering.py

# 2. 召回阶段
python 02_recall/two_tower_pytorch.py

# 3. 粗排阶段
python 03_coarse_ranking/lightweight_models_pytorch.py

# 4. 精排阶段
python 04_ranking/wide_and_deep_pytorch.py
python 04_ranking/deepfm_pytorch.py
python 04_ranking/din_pytorch.py
python 04_ranking/multitask_learning_pytorch.py

# 5. 重排阶段
python 05_reranking/mmr_diversity_pytorch.py

# 6. 混排阶段
python 06_blending/ad_insertion_system.py
```

---

**配套理论文档：**
- [推荐系统基础](../basics.md)
- [深度学习推荐系统学习路径](../deep-learning-recsys-learning-path.md)
