# CS Concepts - 计算机科学概念学习指南

> 系统整理计算机科学的核心概念。每个概念一张导航图，配套代码练习独立管理。

## 🎯 设计哲学

```
cs-concepts/     ← 纯概念导航图：定义 + 公式 + 对比表格 + 关系图 + 交叉引用
practices/       ← 可运行代码：DP题、递归、回溯、面试题、LeetCode题解
```

**每个概念文件力求：**
- 一句话说清本质
- 表格对比核心差异
- ASCII 图展示概念关系
- 交叉链接打通知识网络
- 末尾指向 practices/ 的代码练习

---

## 🏗️ 仓库结构

```
cs-concepts/
├── fundamentals/               # 🏛️ 核心基础概念 ✅ 已压缩
│   ├── algorithms/             # 递归 | DP | 回溯 | 复杂度 | 排序搜索
│   ├── data-structures/        # 数组 | 链表 | 栈队列 | 哈希表 | 树图
│   ├── mathematics/            # 离散数学 | 线性代数 | 概率统计 | 微积分
│   └── programming-concepts/   # 编程范式 | 类型系统 | 并发 | 内存 | 错误处理
│
├── systems/                    # ⚙️ 系统层概念 ✅ 已压缩
│   ├── computer-architecture/  # 组成原理 | 指令集 | 存储 | 流水线 | 性能
│   ├── operating-systems/      # 进程线程 | 同步 | 内存 | 文件 | I/O
│   ├── networks/               # 分层模型 | 物理层→应用层
│   ├── databases/              # 关系模型 | 索引 | 事务 | 存储引擎
│   └── cuda/                   # CUDA 并行计算 (未压缩)
│
├── theory/                     # 🧠 理论基础 ✅ 已压缩
│   ├── complexity-theory/      # P/NP | 近似算法 | 时间空间复杂度
│   └── computation-theory/     # 自动机 | 形式语言 | 可计算性
│
├── languages/                  # 💻 语言核心概念
│   ├── cpp/                    # RAII | 移动语义 | 模板 | 并发 | 系统编程
│   └── go/                     # goroutine | channel | 工程实践
│
├── applications/               # 🚀 应用领域（笔记状态）
│   └── artificial-intelligence/ # ML/DL/推荐系统/TensorFlow
│
└── software-engineering/       # 🏗️ 软件工程

practices/
├── dp/                         # 6道经典DP（爬楼梯→编辑距离）
├── recursion/                  # 6道递归（阶乘→快速幂）
├── backtracking/               # 5道回溯（全排列→N皇后）
├── complexity/                 # 4道复杂度练习
├── sorting/                    # 排序实现
├── leetcode/                   # 400+ 按算法分类的C++/Python题解
├── interviews/                 # 真实面试题（蚂蚁/滴滴/京东/携程等）
├── math/                       # 数学可视化代码（待填）
└── systems/                    # 系统层代码（待填）
```

---

## 📊 学习进度

### ✅ 已完成（概念图 + 练习）
- **算法基础** — 递归 / DP / 回溯 / 复杂度 / 排序搜索
- **数据结构** — 数组链表 / 栈队列 / 哈希表 / 树图
- **数学基础** — 离散 / 线性代数 / 概率统计 / 微积分
- **编程概念** — 范式 / 类型 / 并发 / 内存 / 错误处理
- **操作系统** — 进程线程 / 同步 / 内存 / 文件 / I/O
- **计算机网络** — 分层模型 / 各层协议
- **计算机体系结构** — 组成 / 指令集 / 存储 / 流水线 / 性能
- **数据库** — 关系模型 / 索引 / 事务 / 存储引擎
- **复杂度理论** — P/NP / 近似算法 / 时间复杂度
- **计算理论** — 自动机 / 形式语言 / 可计算性

### 🔄 进行中
- **C++** — 现代 C++（RAII、移动语义、模板、并发）
- **Go** — 基础 / 并发 / 工程实践
- **AI/ML** — 机器学习 / 深度学习 / 推荐系统 / TensorFlow

### 📌 待深入
- NP 完全性、近似算法设计、信息论
- CUDA 并行计算

---

## 📖 使用方式

1. **想学某个概念** → 直接跳到对应的 .md 文件，十分钟读完导航图
2. **想动手练习** → 去 `practices/` 对应目录，跑 `python <文件名>`
3. **想刷题** → `practices/leetcode/` 或 `practices/interviews/`
4. **想查关联** → 每个概念文件末尾都有交叉引用和 `practices/` 链接

---

**开始探索计算机科学的概念世界！** 🚀
