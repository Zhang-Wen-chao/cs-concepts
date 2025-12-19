# TensorFlow 学习笔记

> Google 开源的端到端机器学习平台

## 📚 目录结构

```
tensorflow/
├── README.md                 # 本文件 - TensorFlow 概述
├── 01-core-concepts.md       # 核心概念：张量、计算图、自动微分
├── 02-keras-api.md           # Keras 高级 API
├── 03-data-pipeline.md       # 数据管道 tf.data
├── 04-model-building.md      # 模型构建与训练
└── practices/                # 实践代码
    ├── 01_tensor_basics.py
    ├── 02_mnist_sequential.py
    └── ...
```

## 🎯 学习目标

- [x] 理解 TensorFlow 的核心概念（张量、计算图、自动微分）✅
- [x] 掌握 Keras API 进行快速模型开发 ✅
- [x] 学会使用 tf.data 构建高效数据管道 ✅
- [x] 掌握自定义训练循环、混合精度训练、梯度优化 ✅
- [ ] 实践常见深度学习模型（CNN、RNN 等）
- [ ] 了解模型保存、加载和部署

## 🔗 相关资源

- [官方文档](https://www.tensorflow.org/)
- [TensorFlow 教程](https://www.tensorflow.org/tutorials)
- [Keras API 文档](https://keras.io/)

## 📖 学习路径

### 第一阶段：基础概念 ✅ 已完成
1. ✅ **张量操作** - 理解 TensorFlow 的基本数据结构
2. ✅ **自动微分** - tf.GradientTape 的使用
3. ✅ **计算图** - 静态图 vs 动态图（Eager Execution）

### 第二阶段：Keras API ✅ 已完成
1. ✅ **Sequential API** - 线性堆叠模型
2. ✅ **Functional API** - 复杂模型架构
3. ✅ **Model Subclassing** - 自定义模型类

### 第三阶段：数据管道与训练优化 ✅ 已完成
1. ✅ **tf.data 数据管道** - 高效数据加载与预处理
2. ✅ **自定义训练循环** - 手动控制训练过程
3. ✅ **混合精度训练** - 性能优化
4. ✅ **学习率调度** - Warm-up、余弦衰减等策略

### 第四阶段：实践与部署（待完成）
1. 实践常见深度学习模型
2. 分布式训练
3. 模型部署（TFLite、TF Serving）

## 🔄 与其他内容的关系

- **理论基础**：[深度学习基础](../../deep-learning/)
- **实践应用**：[推荐系统](../../recommendation-systems/)
- **数学基础**：[线性代数](../../../../fundamentals/mathematics/linear-algebra.md)
