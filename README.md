# CS Concepts - 计算机科学概念学习指南

> 系统整理计算机科学的核心概念，用简单易懂的语言解释复杂的CS理论，并记录持续学习的心得

## 🎯 项目简介

这个仓库专注于**概念本身**的解释和理解。概念是计算机科学最精华的部分，掌握了这些概念，你就能在不同领域中灵活运用，拥有真正的洞察力。

**每个概念都力求：**
- 用简单的语言和类比解释
- 配有适当的例子和图表
- 建立概念间的链接关系
- 避免过度的技术实现细节

## 🏗️ 仓库结构

```
cs-concepts/
├── fundamentals/          # 🏛️ 核心基础概念
│   ├── mathematics/      # 数学基础
│   ├── programming-concepts/ # 编程概念
│   ├── data-structures/  # 数据结构
│   └── algorithms/       # 算法基础
│
├── systems/              # ⚙️ 系统层概念
│   ├── computer-architecture/ # 计算机体系结构
│   ├── operating-systems/     # 操作系统
│   ├── networks/             # 网络
│   ├── databases/            # 数据库
│   └── cuda/                 # CUDA 并行计算
│
├── theory/               # 🧠 理论基础
│   ├── computation-theory/   # 计算理论
│   ├── complexity-theory/    # 复杂度理论
│   └── information-theory/   # 信息论
│
├── software-engineering/ # 🏗️ 软件工程概念
│   ├── design-patterns/      # 设计模式
│   ├── design-patterns-practice/ # 设计模式实践
│   ├── software-architecture/ # 软件架构
│   └── programming-methodologies/ # 编程方法论
│
├── courses/              # 📚 课程笔记
│   └── shanghaitech/         # 上海科技大学课程
│
├── languages/            # 💻 编程语言概念
│   ├── language-theory/     # 语言理论
│   ├── comparative-languages/ # 语言对比
│   ├── python/             # Python专题
│   └── cpp/                # C++学习与实践
│
└── applications/         # 🚀 应用领域概念
    ├── artificial-intelligence/ # 人工智能
    ├── web-concepts/           # Web概念
    ├── cryptography/           # 密码学
    └── graphics-multimedia/    # 图形多媒体
```

## 📖 使用方式

- **按需学习**：直接跳转到你感兴趣的概念
- **系统学习**：从基础概念开始，逐层深入
- **概念链接**：通过文档间的链接，理解概念之间的关系
- **查漏补缺**：用作CS知识的参考手册

## ✨ 文档特色

- 每个文档专注一个核心概念
- 用类比和简单例子解释复杂理论
- 大量交叉引用，建立知识网络
- 避免工具使用细节，专注概念理解

## 🔥 当前优先学习主题

- 🔄 **AI/机器学习基础 + 深度学习推荐系统**（开始学习 2025-11-11）
  - 第一阶段：理论学习（已完成）✅
    - ✅ [机器学习核心概念](applications/artificial-intelligence/machine-learning/core-concepts.md) - 监督学习、损失函数、梯度下降
    - 🔄 [神经网络基础](applications/artificial-intelligence/deep-learning/neural-networks.md)
    - 🔄 [推荐系统基础](applications/artificial-intelligence/recommendation-systems/)
  - 第一阶段实践：机器学习基础代码实践（已完成）✅
    - ✅ 01 - 线性回归：从零实现、MSE损失、梯度推导、学习率实验（2025-11-14）
    - ✅ 02 - 梯度下降：BGD/SGD/Mini-batch 对比、学习率影响（2025-11-14）
    - ✅ 03 - 逻辑回归：Sigmoid、交叉熵损失、二分类、决策边界（2025-11-15）
    - ✅ 04 - Softmax 回归：多分类、One-hot编码、与Sigmoid关系（2025-11-15）
    - ✅ 05 - 正则化：L1/L2、过拟合防止、λ参数调优（2025-11-15）
    - ✅ 06 - 神经网络：多层感知机、反向传播、激活函数对比（2025-11-15）
  - 第二阶段：深度学习基础（已完成核心部分）✅
    - ✅ 01 - CNN（卷积神经网络）：卷积、池化、图像分类（2025-11-16）
    - ✅ 02 - RNN/LSTM：循环神经网络、序列建模（2025-11-17）
    - ✅ 03 - Embedding：词嵌入、语义表示、Word2Vec、推荐系统应用（2025-11-19）
    - ✅ 04 - 优化与正则化：Adam、学习率调度、Dropout、BatchNorm（2025-11-17）
    - ✅ 05 - 经典 CNN 架构：LeNet、AlexNet、VGG、ResNet（2025-11-21）
    - ✅ 06 - Attention 机制：Self-Attention、Multi-Head Attention（2025-11-21）
    - ✅ 07 - Transformer：Encoder-Decoder、序列到序列任务（2025-11-22）
    - 目标：为推荐系统打好深度学习基础 ✅ 已达成
  - 第三阶段：深度学习推荐系统（进行中，2025-11-22 开始）
    - ✅ 双塔模型：召回阶段的向量检索（2025-11-22）
    - ✅ Wide & Deep：精排阶段的手工特征交叉（2025-11-22）
    - ✅ DeepFM：精排阶段的自动特征交叉（FM因子分解机）（2025-11-22）
    - ✅ DIN：用户兴趣建模（Attention机制）（2025-11-22）
    - ✅ 多任务学习（MTL）：CTR + CVR 联合训练（2025-11-26）
    - 🔄 粗排、重排、混排（待学习）
    - 目标：掌握完整推荐系统链路
  - → 代码仓库：[machine-learning/practices/](applications/artificial-intelligence/machine-learning/practices/)
  - → [深度学习推荐系统学习路径](applications/artificial-intelligence/recommendation-systems/deep-learning-recsys-learning-path.md)

- 📊 **多模态/图像大模型训练与推理**（<1k卡资源）
  - 以 Megatron 等框架为抓手，纵向打通算子、硬件与推理栈，持续深挖性能边界
  - → [高效训练与部署多模态大模型（<1k卡）](applications/artificial-intelligence/foundation-models/efficient-multimodal-llm-training.md)

- 🔧 **TensorFlow 深度学习框架**（2025-12-17 开始）
  - ✅ 01 - 核心概念：张量、计算图、自动微分、tf.GradientTape（2025-12-17）
  - 🔄 02 - Keras API：Sequential/Functional/Subclassing 三种建模方式（待学习）
  - 目标：掌握 TensorFlow 进行深度学习实践
  - → [TensorFlow 学习笔记](applications/artificial-intelligence/frameworks/tensorflow/)

## 📊 学习进度

> 记录学习历程，见证成长

### 已完成
- ✅ **数学基础** (2025-10)
  - 离散数学
  - 线性代数
  - 概率统计
  - 微积分
- ✅ **编程概念** (2025-10)
  - 编程范式
  - 类型系统
  - 内存管理
  - 并发与并行
  - 抽象与封装
  - 错误处理
- ✅ **数据结构** (2025-10)
  - 数组与列表
  - 栈与队列
  - 哈希表
  - 树与图
- ✅ **算法基础** (2025-10)
  - 复杂度分析
  - 排序与搜索
  - 递归
  - 动态规划
- ✅ **操作系统** (2025-10)
  - 进程与线程
  - 进程同步与互斥
  - 内存管理
  - 文件系统
  - I/O系统
- ✅ **计算机网络** (2025-10)
  - 网络基础与分层模型
  - 物理层与数据链路层
  - 网络层
  - 传输层
  - 应用层
- ✅ **计算机体系结构** (2025-10)
  - 计算机组成原理
  - 指令集架构
  - 存储层次结构
  - 流水线与并行处理
  - 性能评估
- ✅ **数据库** (2025-10)
  - ✅ 数据库基础概念
  - ✅ 索引与查询优化
  - ✅ 关系模型与SQL
  - ✅ 事务与并发控制
  - ✅ 存储引擎

### 进行中
- 🔄 **理论基础** (2025-10 - )
  - ✅ 计算理论
    - ✅ 自动机理论
    - ✅ 形式语言
    - ✅ 可计算性理论
  - ✅ 复杂度理论
    - ✅ 时间与空间复杂度
    - 📖 NP完全性（已浏览，待深入学习）
    - 📖 近似算法（已浏览，待深入学习）
  - 信息论

## 🤔 待深入理解的概念

> 记录当前还不太理解的知识点，定期回顾更新

### 基础概念
- ~~SVD矩阵分解在推荐系统中的应用原理~~ ✅ 已理解（2025-11-22）：矩阵分解 = 权重学习 = Embedding 的数学版本

### 系统层面
- 矩阵乘法的分块优化（Loop Tiling）原理 - `systems/computer-architecture/performance-evaluation.md`
- BLAS 库为何能达到 50 倍性能提升 - `systems/computer-architecture/performance-evaluation.md`

### 理论知识
- AKS素数测试算法详解（第一个确定性多项式时间素数判定算法）- `theory/complexity-theory/np-completeness.md`
- NP完全性理论深入理解（P vs NP, 归约技术，经典NP完全问题）- `theory/complexity-theory/np-completeness.md`
- 最大流-最小割定理（Max-Flow Min-Cut Theorem）及其证明
- 近似算法设计与分析（近似比、PTAS/FPTAS、线性规划舍入）- `theory/complexity-theory/approximation-algorithms.md`

### 感兴趣的主题（未学习）
- 📌 千禧年问题（Millennium Prize Problems）- 克莱数学研究所的七大数学难题 - `fundamentals/mathematics/millennium-problems.md`
  - P vs NP 问题（与计算机科学直接相关）
  - 黎曼猜想、庞加莱猜想等数学问题

### 其他
-

## 📦 未来可能囊括的内容

> 这些内容已添加到仓库，但尚未纳入主体学习规划。可根据实际需要进行整合或清理。

### 课程笔记 (`courses/`)
- 上海科技大学课程材料
  - AI for Science and Engineering
  - Coursera学习资料（机器学习、深度学习专项课程）
  - 编程语言课程（Programming Languages Part A）

### C++实践 (`languages/cpp/`)
- C++多线程编程实践
- LeetCode题解（代码随想录系列）
  - 数组、回溯、二叉树、动态规划等专题

### 设计模式实践 (`software-engineering/design-patterns-practice/`)
- 设计模式的实际应用案例和代码实现

### CUDA并行计算 (`systems/cuda/`)
- CUDA编程基础和并行计算实践

**说明**：这些内容如果与主体规划契合，会逐步整合；如不符合定位，未来可能会移除。

## 💪 重点练习题目

> 经典算法题，展示数据结构的核心应用

### 哈希表必做题 (最优解)
- [ ] **LeetCode 1** - 两数之和：哈希表O(n) vs 暴力O(n²)
- [ ] **LeetCode 3** - 最长无重复子串：哈希表+滑动窗口O(n)
- [ ] **LeetCode 49** - 字母异位词分组：哈希表分组
- [ ] **LeetCode 146** - LRU缓存：哈希表+双向链表O(1)
- [ ] **LeetCode 242** - 有效的字母异位词：哈希表计数

参考：`fundamentals/data-structures/hash-tables.md`

### 树必做题
- [ ] **LeetCode 94/144/145** - 二叉树的中/前/后序遍历：递归和迭代
- [ ] **LeetCode 102** - 二叉树的层序遍历：BFS用队列
- [ ] **LeetCode 104** - 二叉树的最大深度：递归
- [ ] **LeetCode 98** - 验证二叉搜索树：中序遍历有序性
- [ ] **LeetCode 701** - 二叉搜索树插入：BST性质
- [ ] **LeetCode 208** - 实现Trie：字典树基本操作
- [ ] **LeetCode 347** - 前K个高频元素：堆/优先队列
- [ ] **LeetCode 23** - 合并K个有序链表：堆应用

参考：`fundamentals/data-structures/trees-graphs.md`

### 图必做题
- [ ] **LeetCode 200** - 岛屿数量：DFS/BFS基础
- [ ] **LeetCode 207** - 课程表：拓扑排序+环检测
- [ ] **LeetCode 210** - 课程表II：拓扑排序Kahn算法
- [ ] **LeetCode 133** - 克隆图：DFS/BFS + 哈希表
- [ ] **LeetCode 743** - 网络延迟时间：Dijkstra最短路径

参考：`fundamentals/data-structures/trees-graphs.md`

### 排序与搜索必做题
- [ ] **LeetCode 704** - 二分查找：基础二分搜索
- [ ] **LeetCode 34** - 查找第一个和最后一个位置：二分查找边界
- [ ] **LeetCode 153** - 旋转数组最小值：二分搜索变体
- [ ] **LeetCode 912** - 排序数组：实现各种排序算法
- [ ] **LeetCode 347** - 前K个高频元素：Top K问题（堆/快速选择）
- [ ] **LeetCode 215** - 数组中的第K个最大元素：快速选择O(n)
- [ ] **LeetCode 88** - 合并两个有序数组：归并思想

参考：`fundamentals/algorithms/sorting-searching.md`

### 递归必做题
- [ ] **LeetCode 509** - 斐波那契数：递归基础+记忆化
- [ ] **LeetCode 344** - 反转字符串：递归实现
- [ ] **LeetCode 206** - 反转链表：递归vs迭代
- [ ] **LeetCode 50** - Pow(x,n)：快速幂递归
- [ ] **LeetCode 46** - 全排列：回溯算法
- [ ] **LeetCode 78** - 子集：回溯生成所有子集
- [ ] **LeetCode 51** - N皇后：经典回溯问题
- [ ] **LeetCode 22** - 括号生成：回溯+剪枝

参考：`fundamentals/algorithms/recursion.md`

### 动态规划必做题
- [ ] **LeetCode 70** - 爬楼梯：DP入门题
- [ ] **LeetCode 198** - 打家劫舍：状态转移经典题
- [ ] **LeetCode 53** - 最大子数组和：Kadane算法
- [ ] **LeetCode 322** - 零钱兑换：完全背包问题
- [ ] **LeetCode 300** - 最长递增子序列：经典DP O(n²)
- [ ] **LeetCode 1143** - 最长公共子序列：二维DP
- [ ] **LeetCode 72** - 编辑距离：困难但经典
- [ ] **LeetCode 64** - 最小路径和：二维DP入门

参考：`fundamentals/algorithms/dynamic-programming.md`

---

**开始探索计算机科学的概念世界！** 🚀
