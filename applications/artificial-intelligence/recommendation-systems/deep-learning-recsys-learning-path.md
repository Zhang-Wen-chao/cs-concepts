# 深度学习推荐系统学习路径

> 基于 mask 问题分析整理的系统性学习路线，帮助构建推荐系统的概念地图与实践能力  
> 生成时间：2025-10-30

---

## 目录

- [问题诊断](#问题诊断)
- [学习路径总览](#学习路径总览)
- [短期目标（1周）](#短期目标1周)
- [中期目标（1-2个月）](#中期目标1-2个月)
- [长期目标（持续）](#长期目标持续)
- [学习资源](#学习资源)
- [实践项目](#实践项目)
- [检验标准](#检验标准)

---

## 问题诊断

### 这次 mask 问题暴露的知识缺口

```python
# 没能快速识别这段代码的问题：
pred = tf.math.reduce_sum(logits * mask, axis=-1, keepdims=True)  # ❌
```

**根本原因：**

1. ❌ 不知道双塔模型结构 → 看不懂代码在做什么
2. ❌ 不理解训练 vs 推理 → 不知道 mask 不该在 pred 里
3. ❌ 不懂 TensorFlow 计算图 → 不知道为什么 serving 失败
4. ❌ 缺乏调试经验 → 不知道如何定位问题

### 目标

**如果懂模型，应该能做到：**

- ✅ 看到代码 5 秒内识别问题
- ✅ 看到 commit 10 秒内理解原因
- ✅ 30 秒内验证猜测
- ✅ 5 分钟内解决问题

---

## 学习路径总览

```
Level 0: 现状
  └── 不知道双塔结构，不理解模型代码

Level 1: 基础模型理解（1周）
  ├── 推荐系统基础概念
  ├── 双塔模型结构
  ├── 训练 vs 推理区别
  └── 样本加权机制

Level 2: 深度学习模型（1-2个月）
  ├── 经典 CTR 模型
  ├── 多任务学习（MMoE）
  ├── 注意力机制（DIN/DIEN）
  └── TensorFlow 工程实践

Level 3: 进阶与专精（持续）
  ├── 最新模型架构
  ├── 模型优化技巧
  ├── 大规模训练部署
  └── 论文阅读与实现
```

---

## 短期目标（1周）

### Day 1-2: 双塔模型（Two-Tower Model）

**核心概念：**

```
用户特征                物品特征
   ↓                      ↓
用户塔(MLP)            物品塔(MLP)
 512→256→128→64        256→128→64
   ↓                      ↓
用户向量(64维)  ─────×─────  物品向量(64维)
                     ↓
                 相似度分数
```

**学习内容：**

1. 双塔的设计动机（为什么要分两个塔？）
2. 如何训练（point-wise, pair-wise, list-wise）
3. 如何部署（离线向量化 + ANN 检索）
4. 优缺点分析

**必做练习：**

```python
# 用纯 numpy 实现一个最简单的双塔模型
def user_tower(user_features):
    # 实现用户塔
    pass

def item_tower(item_features):
    # 实现物品塔
    pass

def predict(user_vec, item_vec):
    # 计算相似度
    return np.dot(user_vec, item_vec)
```

**学习资源：**

- 📚 论文：《Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations》(YouTube DNN)
- 📺 视频：B 站搜索 "双塔模型"，找推荐量高的
- 💻 代码：看一个简单的双塔实现（DeepCTR 库）

### Day 3: 训练 vs 推理的区别

**核心问题：什么该出现在推理路径？**

```python
# ❌ 错误：mask 混入推理路径
pred = logits * mask  # mask 是训练专用的！
prob = sigmoid(pred)
output = {"score": prob}  # serving 时需要提供 mask

# ✅ 正确：mask 只在 loss 计算
pred = logits  # 推理输出，纯净
loss = mask * cross_entropy(pred, labels)  # mask 只影响 loss
output = {"score": sigmoid(pred)}  # serving 不需要 mask
```

**关键理解：**

| 概念 | 训练时 | 推理时 |
|------|--------|--------|
| Features | ✅ 需要 | ✅ 需要 |
| Labels | ✅ 需要 | ❌ 不需要 |
| Mask/Weights | ✅ 需要（样本加权）| ❌ 不需要 |
| Loss | ✅ 需要（反向传播）| ❌ 不需要 |
| Output | ✅ 有（用于计算 loss）| ✅ 需要（预测结果）|

**必做练习：**

1. 写出一个完整的训练和推理函数，明确区分
2. 画出训练时和推理时的计算图，标出差异
3. 解释为什么 mask 不能出现在 pred 计算中

### Day 4-5: TensorFlow 基础

**核心概念：**

1. **计算图（Graph）**

   ```python
   # 构建图
   a = tf.placeholder(tf.float32)  # 输入节点
   b = tf.placeholder(tf.float32)
   c = a * b  # 计算节点
   d = c + 1

   # 图的依赖链
   d → c → a, b

   # 导出时，从 d 往回追溯，会包含 a 和 b
   ```

2. **Placeholder vs Variable**

   ```python
   # Placeholder: 运行时提供的输入
   features = tf.placeholder(tf.float32, [None, 100])
   labels = tf.placeholder(tf.float32, [None, 1])

   # Variable: 模型参数，训练时更新
   weights = tf.Variable(tf.random.normal([100, 64]))
   bias = tf.Variable(tf.zeros([64]))
   ```

3. **SavedModel 结构**

   ```
   saved_model/
   ├── saved_model.pb        # 计算图定义
   ├── variables/            # 模型参数
   │   ├── variables.data-*
   │   └── variables.index
   └── assets/              # 额外资源
   ```

**必做练习：**

1. 写一个简单的 TensorFlow 模型，保存和加载
2. 用 `saved_model_cli` 查看模型的输入输出
3. 故意在 pred 中加入一个 Placeholder，观察导出后的 signature

**学习资源：**

- 📚 官方文档：TensorFlow 1.x SavedModel
- 💻 实践：保存和加载一个简单模型

### Day 6-7: 样本加权与 mask 机制

**为什么需要样本加权？**

```python
# 场景1: 样本不平衡
# 正样本:负样本 = 1:100
# 需要给正样本更高的权重

mask = tf.where(labels == 1,
                100.0,  # 正样本权重
                1.0)    # 负样本权重
loss = mask * cross_entropy(pred, labels)

# 场景2: 难样本挖掘（Hard Negative Mining）
# 给预测错误的样本更高权重

error = abs(pred - labels)
mask = 1.0 + error  # 错得越多，权重越大
loss = mask * cross_entropy(pred, labels)

# 场景3: 噪声样本处理
# 置信度低的样本降低权重

mask = sample_confidence  # [0, 1]
loss = mask * cross_entropy(pred, labels)
```

**关键：mask 应该在哪里？**

```python
# ✅ 正确：只在 loss 计算
pred = model(features)
loss = mask * loss_fn(pred, labels)

# ❌ 错误：在 pred 计算（污染推理路径）
pred = model(features) * mask
loss = loss_fn(pred, labels)
```

**必做练习：**

1. 实现一个带样本加权的训练循环
2. 对比正确和错误的 mask 使用方式
3. 解释为什么错误方式会导致 serving 失败

---

## 中期目标（1-2个月）

### Week 2-3: 经典 CTR 预估模型

**学习顺序：**

1. **LR（Logistic Regression）** - 最简单
   - 理解特征交叉
   - One-hot 编码

2. **FM（Factorization Machines）** - 二阶交叉
   - 为什么需要 Embedding
   - 内积如何表示交叉

3. **Wide & Deep** - 深度学习入门

   ```
   Wide 部分: 线性模型（记忆）
   Deep 部分: MLP（泛化）
   最后 concat 一起
   ```

4. **DNN** - 纯深度学习
   - 就是把特征 embedding 后接 MLP
   - 你们代码里的基础结构

5. **DeepFM** - FM + DNN
   - FM 替代 Wide 部分
   - 自动学习特征交叉

**学习方法：**

- 📚 每个模型：读论文 → 看代码 → 自己实现
- 💻 使用 DeepCTR 库，理解每个模型的实现
- ✍️ 画出每个模型的结构图

**推荐资源：**

- 💻 DeepCTR 库：https://github.com/shenweichen/DeepCTR
- 📚 《深度学习推荐系统》- 王喆

### Week 4-5: 多任务学习（MMoE）

**你们代码里有这个！**

```python
def MMoE(x):
    # Multi-gate Mixture-of-Experts
    # 多个任务共享底层，各有自己的门控
    pass
```

**核心概念：**

```
             任务1(CVR)    任务2(CTR)
                ↓             ↓
              gate1         gate2
                ↓             ↓
           [ expert1, expert2, expert3 ]
                      ↓
                  共享输入
```

**学习内容：**

1. 为什么需要多任务学习？
2. MMoE vs 传统的 hard parameter sharing
3. Gate 机制如何工作
4. 如何训练和部署

**必做练习：**

1. 实现一个简单的 MMoE
2. 理解你们代码里的 MMoE 是怎么用的
3. 解释为什么有 user MMoE 和 ads MMoE

**学习资源：**

- 📚 论文：《Modeling Task Relationships in Multi-task Learning with MMoE》
- 💻 DeepCTR 的 MMoE 实现

### Week 6-7: 注意力机制（DIN/DIEN）

**如果你们用了 DIN：**

```python
# 用户历史行为序列
user_history = [item1, item2, item3, ...]

# 候选物品
candidate_item = item_x

# DIN: 根据候选物品，对历史行为加权
attention_weights = attention(user_history, candidate_item)
user_interest = weighted_sum(user_history, attention_weights)
```

**学习内容：**

1. 为什么需要注意力机制？
2. Attention 的计算方式
3. DIN vs DIEN 的区别
4. 如何处理变长序列

**必做练习：**

1. 用 numpy 实现 attention 计算
2. 理解为什么要用 target item 做 query
3. 对比有无 attention 的效果差异

### Week 8: LHUC（你们代码里也有！）

```python
def LHUC_NET(x, lhuc_params):
    # Learning Hidden Unit Contribution
    # 为每个神经元学习一个可调节的增益
    pass
```

**核心思想：**

- 在预训练模型基础上，为每个隐层神经元学习一个缩放因子
- 实现模型的快速适配（类似 adapter）

**学习内容：**

1. LHUC 的动机和原理
2. 如何在双塔中使用
3. 与其他迁移学习方法的对比

### TensorFlow 工程实践

**必须掌握的技能：**

1. **模型保存和加载**

   ```python
   # 保存
   tf.saved_model.save(model, export_dir)

   # 加载
   model = tf.saved_model.load(export_dir)

   # 查看 signature
   saved_model_cli show --dir export_dir --all
   ```

2. **计算图分析**

   ```bash
   # 查看 graph 结构
   python -m tensorflow.python.tools.import_pb_to_tensorboard \
     --model_dir=./model \
     --log_dir=./logs

   # 用 TensorBoard 可视化
   tensorboard --logdir=./logs
   ```

3. **模型调试**

   ```python
   # 打印所有 ops
   for op in graph.get_operations():
       print(op.name, op.type)

   # 查找特定节点
   graph.get_tensor_by_name("dense/mul_1:0")

   # 查看依赖
   for input_tensor in op.inputs:
       print(input_tensor.name)
   ```

4. **TensorFlow Serving**

   ```bash
   # 启动 serving
   tensorflow_model_server \
     --model_base_path=/models/my_model \
     --rest_api_port=8501

   # 发送请求
   curl -X POST http://localhost:8501/v1/models/my_model:predict \
     -d '{"instances": [...]}'
   ```

**实践项目：**

1. 完整走通训练 → 导出 → serving 流程
2. 故意制造一些错误，学会调试
3. 对比不同导出方式的差异

---

## 长期目标（持续）

### 最新模型架构

**跟进行业发展：**

1. Transformer 在推荐中的应用（BST, BERT4Rec）
2. 对比学习（Contrastive Learning）
3. 自监督学习
4. 大模型在推荐中的应用

**学习方法：**

- 📚 关注顶会论文（RecSys, KDD, WWW, SIGIR）
- 💻 复现经典论文
- 🎯 参加 Kaggle 竞赛

### 模型优化技巧

**工程优化：**

1. 模型压缩（pruning, quantization, distillation）
2. 在线学习与增量更新
3. 特征工程自动化
4. 超参数调优

**系统优化：**

1. 分布式训练（Parameter Server, Ring AllReduce）
2. GPU 加速技巧
3. 混合精度训练
4. 模型并行 vs 数据并行

### 推荐系统全栈

**不只是模型：**

1. 召回策略（多路召回、级联召回）
2. 排序模型（粗排、精排、重排）
3. 在线实验（A/B 测试）
4. 效果评估（线上指标 vs 线下指标）
5. 冷启动问题
6. 实时性要求

---

## 学习资源

### 书籍

**入门级：**

1. 📘 《推荐系统实践》- 项亮
   - 最适合入门
   - 偏工程实践

2. 📘 《深度学习推荐系统》- 王喆
   - 很全面，从经典到深度学习
   - 有代码示例

**进阶级：**

3. 📘 《Recommender Systems Handbook》
   - 经典教材，很全面
   - 理论较多

4. 📘 《深度学习》- Ian Goodfellow
   - 深度学习基础
   - 数学较多

### 在线课程

1. 🎥 吴恩达《机器学习》
   - Coursera 免费
   - 打基础必看

2. 🎥 李宏毅《深度学习》
   - YouTube 免费
   - 讲得很清楚

3. 🎥 斯坦福《CS224N》
   - NLP 课程，但很多技术通用
   - Attention 机制讲得好

### 代码库

1. 💻 **DeepCTR**
   - https://github.com/shenweichen/DeepCTR
   - 各种 CTR 模型的标准实现
   - **强烈推荐，必看！**

2. 💻 **TensorFlow Recommenders**
   - Google 官方推荐库
   - 工程化较好

3. 💻 **RecBole**
   - 推荐系统研究工具
   - 学术界常用

### 论文阅读

**经典必读：**

1. Wide & Deep Learning (Google, 2016)
2. DeepFM (HIT, 2017)
3. DIN (Alibaba, 2018)
4. MMoE (Google, 2018)
5. DIEN (Alibaba, 2019)

**论文阅读技巧：**

1. 先读摘要和结论，判断是否值得细读
2. 重点看模型结构图
3. 对比 baseline，理解改进点
4. 如果有代码，一定要看

### 社区资源

1. **知乎专栏** - 搜 "推荐系统"
2. **微信公众号** - 机器学习算法与自然语言处理
3. **GitHub Awesome Lists** - awesome-deep-learning, awesome-recommendation
4. **arXiv** - 最新论文

---

## 实践项目

### 项目 1: 实现一个完整的双塔模型（1周）

**目标：**

从零实现训练和部署，深入理解每个环节

**步骤：**

1. 数据准备（MovieLens 或自己构造）
2. 特征工程（user features, item features）
3. 模型定义（user tower, item tower）
4. 训练循环（loss, optimizer, metrics）
5. 模型导出（SavedModel 格式）
6. 离线评估（AUC, 召回率）
7. Serving 部署（TF Serving 或自己写）

**检验标准：**

- ✅ 模型能正常训练，loss 下降
- ✅ 导出的模型只包含 features 输入，没有 labels
- ✅ Serving 能正常响应，返回预测分数
- ✅ 能解释每一行代码的作用

### 项目 2: 复现你们的模型结构（2周）

**目标：**

完全理解你们的代码，能自己写出来

**步骤：**

1. 画出完整的模型结构图
2. 理解每个组件（MMoE, LHUC, DIN 等）
3. 用纯 TensorFlow 重新实现
4. 对比你的实现和原代码的差异
5. 理解训练时的每个 trick

**检验标准：**

- ✅ 能画出清晰的架构图
- ✅ 能向别人讲解每个模块的作用
- ✅ 能指出可能的改进点
- ✅ 遇到问题知道去哪里 debug

### 项目 3: 参加一次 Kaggle 竞赛

**推荐竞赛：**

- CTR 预估相关
- 推荐系统相关

**目标：**

- 不是拿名次，而是学习完整的建模流程
- 看 top solution，学习高手的思路

---

## 检验标准

### 短期检验（1周后）

**能回答这些问题：**

1. 什么是双塔模型？为什么要用双塔？
2. 训练和推理有什么区别？
3. Placeholder 和 Variable 的区别？
4. 为什么 mask 不能出现在 pred 计算中？
5. 如何查看一个 SavedModel 的输入输出？

**能做到这些：**

1. ✅ 看到双塔代码，5 秒内理解在做什么
2. ✅ 看到 `pred = logits * mask`，立刻知道有问题
3. ✅ 用 `saved_model_cli` 查看模型 signature
4. ✅ 实现一个最简单的双塔模型

### 中期检验（2 个月后）

**能回答这些问题：**

1. 常见的 CTR 模型有哪些？各有什么特点？
2. MMoE 解决什么问题？和单任务有什么区别？
3. 注意力机制在推荐中怎么用？
4. 如何 debug 一个 TensorFlow 模型？

**能做到这些：**

1. ✅ 完整实现一个 CTR 模型（包括训练和部署）
2. ✅ 看懂你们的完整模型代码
3. ✅ 遇到 serving 问题，能快速定位
4. ✅ 能提出模型改进建议

### 长期检验（持续）

**成为一个合格的算法工程师：**

1. ✅ 能独立设计和实现推荐模型
2. ✅ 能读懂并复现论文
3. ✅ 能分析和优化线上效果
4. ✅ 能解决各种工程问题
5. ✅ 能指导新人成长

---

## 学习建议

### 学习方法

**1. 主动学习，不要被动接受**

```
❌ 只看视频、只看书
✅ 看完立刻实践，写代码验证
```

**2. 带着问题学习**

```
❌ 泛泛地看资料
✅ 明确要解决什么问题，针对性学习
```

**3. 费曼学习法**

```
学习一个概念 → 用简单语言讲给别人听
如果讲不清楚 → 说明还没真正理解
```

**4. 构建知识体系**

```
❌ 碎片化学习，东一块西一块
✅ 系统化学习，形成知识网络
```

### 时间分配

**每天 2-3 小时：**

- 40% 看书 / 看视频（理论学习）
- 40% 写代码（实践）
- 20% 总结和思考（写笔记、画图）

**周末：**

- 做一个完整的小项目
- 复盘一周的学习
- 查漏补缺

### 避免的坑

**1. 只看不练**

- 看再多资料，不写代码等于零

**2. 追求完美**

- 不要想着 "学完再实践"
- 边学边练，快速迭代

**3. 贪多求全**

- 不要想着 "把所有模型都学一遍"
- 先把基础打牢，再拓展

**4. 闭门造车**

- 多交流，多请教
- 看别人的代码，学习别人的思路

---

## 总结

### 这次 mask 问题的核心教训

**你缺的不是智商，而是系统性的基础知识：**

1. 模型结构（双塔）
2. 训练原理（样本加权）
3. 工程实践（计算图、serving）

**补上这些，下次 5 分钟就能解决。**

### 学习是一个过程

```
现在: 不懂双塔，看不懂代码
  ↓ (1 周系统学习)
1 周后: 理解双塔，能看懂基本代码
  ↓ (1 个月系统学习)
1 个月后: 理解大部分模型，能实现简单的
  ↓ (2 个月系统学习)
2 个月后: 能独立设计和实现，能快速定位问题
  ↓ (持续学习)
半年后: 成为这个领域的专家
```

### 开始行动

**今天就开始：**

1. 找一篇双塔模型的博客或视频，花 1 小时看完
2. 找 DeepCTR 的双塔实现，花 1 小时读代码
3. 自己用 numpy 实现一个最简单的双塔，花 2 小时

**不要拖延，现在就开始！**

---

*"The expert in anything was once a beginner."*

— Helen Hayes

祝学习顺利！💪

