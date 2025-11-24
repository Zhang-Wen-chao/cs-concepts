# Artificial Intelligence - 人工智能

> 让机器具备智能行为的理论、方法和应用

## 📚 主题结构

每个主题包含**理论文档**和**实践代码**（practices/ 子文件夹）

### [机器学习 (Machine Learning)](machine-learning/)
从数据中自动学习规律的方法

**完整内容：** [machine-learning/README.md](machine-learning/README.md) - 包含所有模块说明

**实践代码：** [machine-learning/practices/](machine-learning/practices/)
- ✅ 线性回归、梯度下降、逻辑回归
- ✅ Softmax 回归、正则化
- ✅ 神经网络基础（多层感知机）

---

### [深度学习 (Deep Learning)](deep-learning/)
基于神经网络的深层学习方法

**完整内容：** [deep-learning/README.md](deep-learning/README.md) - 包含所有模块说明和学习路径

**实践代码：** [deep-learning/practices/](deep-learning/practices/)
- ✅ CNN（卷积神经网络）- 图像处理
- ✅ RNN/LSTM（循环神经网络）- 序列数据
- ✅ Embedding 技术 - 词嵌入与表示学习
- ✅ 优化与正则化 - Adam、Dropout、BatchNorm

---

### [推荐系统 (Recommendation Systems)](recommendation-systems/)
结合工程与建模的智能推荐体系

**理论文档：**
- [推荐系统基础](recommendation-systems/basics.md)
- [深度学习推荐系统学习路径](recommendation-systems/deep-learning-recsys-learning-path.md) - 从双塔入手的系统性学习路径

**实践代码：** [recommendation-systems/practices/](recommendation-systems/practices/)
- ✅ 协同过滤（User-based、Item-based、矩阵分解）
- ✅ 双塔模型（Two-Tower Model）- 召回阶段（2025-11-22）
- ✅ Wide & Deep - 精排阶段，手工交叉特征（2025-11-22）
- ✅ DeepFM - 精排阶段，自动交叉特征（2025-11-22）
- 🔄 DIN（Deep Interest Network）- 用户兴趣建模（待学习）

---

### [基础模型与大模型 (Foundation Models)](foundation-models/)
面向大规模通用模型的系统化训练与部署

**理论文档：**
- [高效训练与部署多模态大模型（<1k卡）](foundation-models/efficient-multimodal-llm-training.md) - 在受限资源下构建具备竞争力的多模态/图像大模型训练与推理体系

---

## 🚀 快速开始

### 环境配置

```bash
# 安装依赖
cd applications/artificial-intelligence
pip install -r requirements.txt

# 运行实践代码（以机器学习为例）
cd machine-learning/practices
python 01_linear_regression.py
```

### 学习路径

1. **机器学习基础** → [machine-learning/practices/](machine-learning/practices/)
2. **深度学习基础** → [deep-learning/practices/](deep-learning/practices/)
3. **推荐系统** → [recommendation-systems/practices/](recommendation-systems/practices/)

---

## 📖 学习建议

- **先看理论文档**，理解概念
- **运行实践代码**，观察效果
- **修改参数实验**，深入理解
- **对比标准实现**，验证正确性

---

人工智能是一个快速发展的交叉学科，结合了数学、统计学、计算机科学和认知科学的方法。
