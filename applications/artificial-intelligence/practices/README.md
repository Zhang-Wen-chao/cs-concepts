# AI/ML 实践代码

> 配合理论文档的代码实践，通过动手写代码来深入理解机器学习和深度学习概念

## 📁 目录结构

```
practices/
├── ml-basics/              # 机器学习基础实践
│   ├── 01_linear_regression.ipynb
│   ├── 02_gradient_descent.ipynb
│   └── 03_logistic_regression.ipynb
├── neural-networks/        # 神经网络实践
│   ├── 01_perceptron.ipynb
│   └── 02_backpropagation.ipynb
├── recommendation/         # 推荐系统实践
│   ├── 01_collaborative_filtering.ipynb
│   └── 02_two_tower_model.ipynb
└── data/                   # 实验数据（被 .gitignore 排除）
```

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter
jupyter notebook
```

### 学习顺序

1. **机器学习基础** (`ml-basics/`)
   - 理解监督学习的基本流程
   - 掌握梯度下降原理
   - 实现简单的分类和回归

2. **神经网络** (`neural-networks/`)
   - 从感知机到多层网络
   - 理解反向传播算法
   - 实现简单的深度学习模型

3. **推荐系统** (`recommendation/`)
   - 协同过滤基础
   - 双塔模型实现
   - CTR 预估模型

## 📚 配套理论文档

- [机器学习核心概念](../machine-learning/core-concepts.md)
- [神经网络基础](../deep-learning/neural-networks.md)
- [推荐系统基础](../recommendation-systems/basics.md)
- [深度学习推荐系统学习路径](../recommendation-systems/deep-learning-recsys-learning-path.md)

## 💡 学习建议

### 边学边练的方式

1. **先看理论文档** - 理解概念
2. **运行代码** - 看实际效果
3. **修改参数** - 观察变化
4. **自己实现** - 深入理解
5. **记录笔记** - 总结心得

### 实践原则

- ✅ 从最简单的例子开始
- ✅ 每个概念都写代码验证
- ✅ 多做实验，观察现象
- ✅ 不要只复制粘贴，理解每一行
- ❌ 不要追求完美，快速迭代

## 🛠️ 工具推荐

- **Jupyter Notebook** - 交互式编程，方便实验
- **NumPy** - 数值计算基础
- **Matplotlib** - 可视化结果
- **scikit-learn** - 对比标准实现
- **TensorFlow/PyTorch** - 深度学习框架

## 📝 代码规范

- 每个 notebook 包含完整的说明和注释
- 代码尽量简洁，突出核心概念
- 包含可视化，帮助理解
- 提供练习题，巩固知识

## 🎯 学习目标

通过这些实践代码，你应该能够：

- ✅ 理解机器学习的基本原理
- ✅ 掌握神经网络的训练过程
- ✅ 实现简单的推荐系统模型
- ✅ 具备调试和优化模型的能力
- ✅ 为工作中的实际问题打下基础

---

**记住：看懂代码不算懂，写出来才是真的懂！** 💪
