# 深度学习基础实践

> 从基础神经网络到现代深度学习架构

## 💡 学习方式

**每个模块都包含两个版本：**
- 📝 **NumPy版本** - 手写实现，理解数学原理（无GPU）
- 🚀 **PyTorch版本** - 框架实现，实际应用，GPU加速

---

## 📚 学习内容

### 第一阶段：基础架构（已完成 - 双版本）✅

#### 01 - 卷积神经网络 (CNN) ✅
**概念：** 专门处理网格结构数据（如图像）的神经网络

**实现内容：**
- 卷积层（Convolution）：特征提取
- 池化层（Pooling）：降维
- 经典架构：LeNet、简化版 ResNet
- 图像分类实战（MNIST）

**文件：**
- `01_cnn_basics_numpy.py` - NumPy手写实现
- `01_cnn_basics_pytorch.py` - PyTorch实现 + GPU训练 ✅

**已掌握：** 2025-11-17（双版本完成）

---

#### 02 - 循环神经网络 (RNN/LSTM) ✅
**概念：** 处理序列数据的神经网络

**实现内容：**
- RNN 基础：循环结构
- LSTM：长短期记忆
- 序列预测、文本生成
- 时间序列分析

**文件：**
- `02_rnn_lstm_numpy.py` - NumPy手写实现
- `02_rnn_lstm_pytorch.py` - PyTorch实现 + GPU训练 ✅

**已掌握：** 2025-11-17（双版本完成）

---

#### 03 - Embedding 技术 ✅
**概念：** 将离散符号映射到连续向量空间

**实现内容：**
- Word Embedding：Word2Vec、GloVe
- Item Embedding：推荐系统基础
- 语义相似度计算
- 可视化降维（PCA）

**文件：**
- `03_embedding_numpy.py` - NumPy手写实现
- `03_embedding_pytorch.py` - PyTorch实现（nn.Embedding + GPU）✅

**已掌握：** 2025-11-17（双版本完成）

---

#### 04 - 优化与正则化技术 ✅
**概念：** 训练深度网络的技巧

**实现内容：**
- 优化器：SGD、Momentum、Adam、AdamW
- 学习率调度：StepLR、CosineAnnealing、OneCycleLR
- 正则化：Dropout、BatchNorm、Weight Decay
- 梯度裁剪、权重初始化

**文件：**
- `04_optimization_regularization_numpy.py` - NumPy手写实现
- `04_optimization_regularization_pytorch.py` - PyTorch完整实践 ✅

**已掌握：** 2025-11-17（双版本完成）

---

### 第二阶段：现代架构（核心）⭐

#### 05 - Attention 机制
**概念：** 让模型关注重要信息（Transformer的基础）

**实现内容：**
- Self-Attention：查询、键、值
- Multi-Head Attention：多头注意力
- Scaled Dot-Product Attention
- 注意力可视化

**文件：**
- `05_attention.py` - NumPy手写实现
- `05_attention_pytorch.py` - PyTorch实现

**为什么重要？**
- Transformer的核心组件
- BERT、GPT、ViT都基于此
- 现代NLP/CV的基础

---

#### 06 - Transformer 架构
**概念：** "Attention is All You Need"（现代深度学习基石）

**实现内容：**
- Encoder-Decoder结构
- Position Encoding：位置编码
- Layer Normalization
- 机器翻译实战

**文件：**
- `06_transformer.py` - NumPy简化实现
- `06_transformer_pytorch.py` - PyTorch完整实现

**为什么重要？**
- 取代RNN成为序列建模主流
- BERT、GPT的基础
- 视觉Transformer（ViT）

---

#### 07 - 预训练与微调
**概念：** 迁移学习实践（站在巨人肩膀上）

**实现内容：**
- 加载预训练模型（ResNet、BERT）
- 冻结层与解冻
- 微调策略
- 特征提取 vs 全模型微调

**文件：**
- `07_transfer_learning_pytorch.py` - PyTorch实现（无NumPy版）

**为什么重要？**
- 工业界标准做法
- 小数据也能训练好模型
- 节省计算资源

---

### 第三阶段：计算机视觉（高级）

#### 08 - 目标检测
**概念：** 检测图像中的多个物体（位置 + 类别）

**实现内容：**
- YOLO：You Only Look Once
- Anchor Box
- Non-Maximum Suppression (NMS)
- 实战：检测图片中的物体

**文件：**
- `08_object_detection_pytorch.py` - PyTorch实现

---

#### 09 - 图像分割
**概念：** 像素级分类（每个像素属于哪个类别）

**实现内容：**
- U-Net：医学图像分割
- 语义分割 vs 实例分割
- FCN（全卷积网络）

**文件：**
- `09_image_segmentation_pytorch.py` - PyTorch实现

---

#### 10 - 视觉 Transformer (ViT)
**概念：** 用Transformer处理图像

**实现内容：**
- Patch Embedding
- Vision Transformer架构
- 与CNN对比

**文件：**
- `10_vision_transformer_pytorch.py` - PyTorch实现

---

### 第四阶段：生成模型

#### 11 - GAN (生成对抗网络)
**概念：** 生成器 vs 判别器（对抗训练）

**实现内容：**
- GAN原理
- 生成手写数字（MNIST）
- DCGAN（深度卷积GAN）
- 训练稳定性技巧

**文件：**
- `11_gan.py` - NumPy简化实现
- `11_gan_pytorch.py` - PyTorch实现

---

#### 12 - VAE (变分自编码器)
**概念：** 概率生成模型

**实现内容：**
- 编码器-解码器
- 重参数化技巧
- 潜在空间可视化

**文件：**
- `12_vae_pytorch.py` - PyTorch实现

---

#### 13 - Diffusion Models
**概念：** 扩散模型（Stable Diffusion、DALL-E的基础）

**实现内容：**
- 前向扩散过程
- 反向去噪过程
- DDPM（去噪扩散概率模型）

**文件：**
- `13_diffusion_pytorch.py` - PyTorch实现

---

### 第五阶段：高级主题（可选）

#### 14 - 强化学习基础
**概念：** 智能体与环境交互学习

**实现内容：**
- DQN（深度Q网络）
- Policy Gradient
- 实战：玩游戏

**文件：**
- `14_reinforcement_learning_pytorch.py` - PyTorch实现

---

#### 15 - 图神经网络 (GNN)
**概念：** 处理图结构数据

**实现内容：**
- 图卷积网络（GCN）
- 消息传递机制
- 节点分类、图分类

**文件：**
- `15_gnn_pytorch.py` - PyTorch实现

---

#### 16 - 多模态学习
**概念：** 融合多种模态（图像+文本）

**实现内容：**
- CLIP：对比学习
- 图像-文本匹配
- 多模态Transformer

**文件：**
- `16_multimodal_pytorch.py` - PyTorch实现

---

## 🎯 学习目标

通过这些实践，你应该能够：

### 第一阶段（已完成）✅
- ✅ 理解 CNN 处理图像的原理
- ✅ 掌握 RNN/LSTM 处理序列的方法
- ✅ 理解 Embedding 的作用（推荐系统核心）
- ✅ 掌握深度网络训练技巧

### 第二阶段（核心）⭐
- 🎯 掌握 Attention 机制（现代DL基础）
- 🎯 理解 Transformer 架构
- 🎯 会用预训练模型（工业界必备）

### 第三阶段（CV高级）
- 🎯 实现目标检测（YOLO）
- 🎯 实现图像分割（U-Net）
- 🎯 理解视觉Transformer

### 第四阶段（生成模型）
- 🎯 理解GAN的对抗训练
- 🎯 掌握VAE的生成原理
- 🎯 了解Diffusion模型

---

## 📖 学习路径建议

### 推荐路径（直奔推荐系统）
```
第一阶段（已完成）
    ↓
05 - Attention（必学！）
    ↓
06 - Transformer（必学！）
    ↓
推荐系统实践（双塔模型）
    ↓
（需要时回来学其他模块）
```

### 完整路径（系统学习深度学习）
```
第一阶段 → 第二阶段 → 第三阶段 → 第四阶段 → 第五阶段
（4个模块）  （3个模块）  （3个模块）  （3个模块）  （3个模块）
```

### CV方向路径
```
第一阶段 → 第二阶段 → 第三阶段 → 生成模型
(CNN+优化) (Attention) (检测+分割) (GAN+Diffusion)
```

### NLP方向路径
```
第一阶段 → 第二阶段 → 预训练模型应用
(RNN+Embedding) (Attention+Transformer) (BERT/GPT微调)
```

---

## 💡 学习建议

### NumPy vs PyTorch
- **NumPy版本：** 理解原理，手写反向传播
- **PyTorch版本：** 实际应用，GPU加速，工业实践

### 学习顺序
1. **先看NumPy版本** - 理解数学原理
2. **再看PyTorch版本** - 学习框架使用
3. **对比两个版本** - 理解框架为你做了什么

### GPU训练
```python
# 检查GPU
import torch
print(torch.cuda.is_available())  # True表示有GPU

# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

---

## 🚀 下一步

### 当前进度
- ✅ 第一阶段完成（4个模块，NumPy + PyTorch 双版本）
- ✅ PyTorch版本全部完成（01-04）- GPU加速训练
- 📍 准备学习第二阶段（Attention + Transformer）

### 建议行动
1. **继续推荐系统** - 先学双塔模型（工作相关）
2. **回来补Attention** - 对理解推荐系统有帮助
3. **实践PyTorch版本** - 在GPU上运行，体验加速效果

