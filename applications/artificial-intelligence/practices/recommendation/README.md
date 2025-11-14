# 推荐系统实践

> 从协同过滤到深度学习推荐模型

## 📖 学习内容

### 01 - 协同过滤 (Collaborative Filtering)
**概念：** 基于用户行为的推荐

**实现内容：**
- 用户-物品评分矩阵
- 基于用户的协同过滤 (User-based CF)
- 基于物品的协同过滤 (Item-based CF)
- 矩阵分解 (Matrix Factorization)

**配套文档：** [推荐系统基础](../../recommendation-systems/basics.md)

---

### 02 - 双塔模型 (Two-Tower Model)
**概念：** 深度学习推荐的基础架构

**实现内容：**
- 用户塔和物品塔
- Embedding 层
- 向量相似度计算
- 训练和推理分离
- **重点：理解 mask 应该在哪里**

**配套文档：** [深度学习推荐系统学习路径](../../recommendation-systems/deep-learning-recsys-learning-path.md)

---

## 🎯 学习目标

通过这些练习，你应该能够：
- ✅ 理解推荐系统的基本原理
- ✅ 掌握双塔模型的结构
- ✅ 理解训练和推理的区别
- ✅ 能够解决实际的 serving 问题

## 💡 学习建议

1. **从简单数据开始** - 用 MovieLens 等公开数据集
2. **理解分离** - 训练时的逻辑 vs 推理时的逻辑
3. **关注细节** - mask、loss、输出的位置很关键
4. **对比工作代码** - 理解你们项目的模型结构

---

**这部分直接关系到你的工作，好好学！** 💼
