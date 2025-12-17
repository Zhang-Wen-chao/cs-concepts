# TensorFlow 核心概念

> 理解 TensorFlow 的基础：张量、计算图、自动微分

## 🎯 核心概念总览

TensorFlow 的名字来源于 **Tensor（张量）** + **Flow（流动）**，描述了数据（张量）在计算图中流动的过程。

```
输入数据（张量） → 计算图（操作序列） → 输出结果（张量）
                        ↑
                   自动微分（反向传播）
```

## 1. 张量（Tensor）

### 什么是张量？

张量是 TensorFlow 中的基本数据结构，可以理解为**多维数组**：

```python
import tensorflow as tf

# 0维张量：标量
scalar = tf.constant(42)          # shape: ()

# 1维张量：向量
vector = tf.constant([1, 2, 3])   # shape: (3,)

# 2维张量：矩阵
matrix = tf.constant([[1, 2],
                      [3, 4]])     # shape: (2, 2)

# 3维张量：如RGB图像
image = tf.zeros([256, 256, 3])   # shape: (256, 256, 3)

# 4维张量：一批图像
batch = tf.zeros([32, 256, 256, 3])  # shape: (batch, height, width, channels)
```

### 张量的维度类比

| 维度 | 数学名称 | 例子 | 形状 |
|------|---------|------|------|
| 0D | 标量（Scalar） | 温度: 36.5°C | `()` |
| 1D | 向量（Vector） | 时间序列: [1, 2, 3, 4] | `(4,)` |
| 2D | 矩阵（Matrix） | 灰度图像: 28×28 | `(28, 28)` |
| 3D | 3阶张量 | RGB图像: 256×256×3 | `(256, 256, 3)` |
| 4D | 4阶张量 | 图像批次: 32×256×256×3 | `(32, 256, 256, 3)` |

### 张量的属性

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

print(tensor.shape)     # TensorShape([2, 3]) - 形状
print(tensor.dtype)     # tf.int32 - 数据类型
print(tensor.numpy())   # 转换为 NumPy 数组
```

### 张量操作

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# 基本运算
c = a + b               # [5, 7, 9] - 逐元素相加
d = a * b               # [4, 10, 18] - 逐元素相乘
e = tf.matmul(a, b)     # 矩阵乘法（需要形状匹配）

# 形状操作
x = tf.constant([[1, 2], [3, 4]])
y = tf.reshape(x, [4])              # [1, 2, 3, 4] - 重塑
z = tf.transpose(x)                 # [[1, 3], [2, 4]] - 转置

# 聚合操作
mean = tf.reduce_mean(a)            # 2.0 - 平均值
sum_val = tf.reduce_sum(a)          # 6 - 求和
max_val = tf.reduce_max(a)          # 3 - 最大值
```

## 2. 计算图（Computation Graph）

### 静态图 vs 动态图

TensorFlow 有两种执行模式：

#### TensorFlow 1.x：静态图（Graph Mode）

```python
# 旧版本：先定义图，再运行
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 第一步：定义计算图
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a + b

# 第二步：创建会话并执行
with tf.Session() as sess:
    result = sess.run(c, feed_dict={a: 3.0, b: 4.0})
    print(result)  # 7.0
```

**特点**：
- ✅ 性能优化好（编译时优化）
- ✅ 适合生产部署
- ❌ 调试困难
- ❌ 代码不直观

#### TensorFlow 2.x：动态图（Eager Execution）

```python
# 新版本：默认启用 Eager Execution（即时执行）
import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(4.0)
c = a + b
print(c.numpy())  # 7.0 - 立即得到结果
```

**特点**：
- ✅ 代码简洁，类似 NumPy
- ✅ 调试方便（可以用 print、断点）
- ✅ 更 Pythonic
- ⚠️ 性能略低于静态图

### 兼顾性能：@tf.function

使用 `@tf.function` 将 Python 函数编译为静态图，兼顾易用性和性能：

```python
import tensorflow as tf
import time

# 普通 Python 函数（慢）
def slow_function(x, y):
    return x ** 2 + y ** 2

# 使用 @tf.function 装饰（快）
@tf.function
def fast_function(x, y):
    return x ** 2 + y ** 2

x = tf.constant(3.0)
y = tf.constant(4.0)

# 性能对比
start = time.time()
for _ in range(10000):
    slow_function(x, y)
print(f"普通函数耗时: {time.time() - start:.4f}s")

start = time.time()
for _ in range(10000):
    fast_function(x, y)
print(f"@tf.function 耗时: {time.time() - start:.4f}s")
```

**最佳实践**：
- 开发调试时：使用 Eager Execution
- 训练模型时：使用 `@tf.function` 加速
- 生产部署时：使用 SavedModel 格式（自动静态图）

## 3. 自动微分（Automatic Differentiation）

### 为什么需要自动微分？

深度学习的核心是**梯度下降**，需要计算损失函数对参数的梯度：

```
θ_new = θ_old - learning_rate * ∇L(θ)
                                  ↑
                           需要自动计算这个梯度
```

### tf.GradientTape：自动微分的核心

`tf.GradientTape` 是 TensorFlow 的"录音机"，记录计算过程，然后自动计算梯度。

#### 基本用法

```python
import tensorflow as tf

# 定义变量
x = tf.Variable(3.0)

# 使用 GradientTape 记录计算过程
with tf.GradientTape() as tape:
    y = x ** 2  # y = x²

# 计算梯度 dy/dx = 2x
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())  # 6.0 (因为 2 * 3 = 6)
```

#### 多变量梯度

```python
# 线性回归例子: y = wx + b
w = tf.Variable(2.0)
b = tf.Variable(1.0)
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    y = w * x + b  # y = 2*3 + 1 = 7
    loss = y ** 2   # loss = 49

# 计算梯度
gradients = tape.gradient(loss, [w, b])
dL_dw, dL_db = gradients

print(f"∂Loss/∂w = {dL_dw.numpy()}")  # 42.0
print(f"∂Loss/∂b = {dL_db.numpy()}")  # 14.0
```

#### 持久性 GradientTape

默认情况下，`GradientTape` 只能调用一次 `.gradient()`。如需多次计算，使用 `persistent=True`：

```python
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y = x ** 2
    z = y ** 2

# 多次计算梯度
dy_dx = tape.gradient(y, x)  # dy/dx = 2x = 6
dz_dx = tape.gradient(z, x)  # dz/dx = 4x³ = 108

print(dy_dx.numpy())  # 6.0
print(dz_dx.numpy())  # 108.0

del tape  # 手动删除以释放资源
```

#### 监视常量（watch）

`GradientTape` 默认只监视 `tf.Variable`。要监视常量，需要显式调用 `tape.watch()`：

```python
x = tf.constant(3.0)  # 常量，默认不监视

with tf.GradientTape() as tape:
    tape.watch(x)  # 显式监视
    y = x ** 2

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())  # 6.0
```

### 实战：手写梯度下降

```python
import tensorflow as tf

# 目标：找到 y = x² 的最小值点（答案：x = 0）

x = tf.Variable(10.0)  # 初始值
learning_rate = 0.1

for step in range(50):
    with tf.GradientTape() as tape:
        y = x ** 2  # 目标函数

    # 计算梯度
    dy_dx = tape.gradient(y, x)

    # 梯度下降更新
    x.assign(x - learning_rate * dy_dx)

    if step % 10 == 0:
        print(f"Step {step}: x = {x.numpy():.4f}, y = {y.numpy():.4f}")

# 输出：
# Step 0: x = 8.0000, y = 100.0000
# Step 10: x = 2.6843, y = 7.2056
# Step 20: x = 0.9005, y = 0.8109
# Step 30: x = 0.3021, y = 0.0913
# Step 40: x = 0.1013, y = 0.0103
```

## 4. 变量（Variable）

### Variable vs Tensor

| 特性 | tf.Tensor | tf.Variable |
|------|----------|-------------|
| 可变性 | 不可变 | 可变 |
| 梯度追踪 | 需要 watch() | 自动追踪 |
| 用途 | 数据、中间结果 | 模型参数（权重、偏置） |

```python
# Tensor：不可变
t = tf.constant([1, 2, 3])
# t[0] = 10  # 错误！不能修改

# Variable：可变
v = tf.Variable([1, 2, 3])
v[0].assign(10)  # 正确！
print(v.numpy())  # [10, 2, 3]
```

### Variable 的操作

```python
w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])

# 赋值操作
w.assign([[0.0, 0.0], [0.0, 0.0]])       # 完全替换
w.assign_add([[1.0, 1.0], [1.0, 1.0]])   # 加法赋值
w.assign_sub([[0.5, 0.5], [0.5, 0.5]])   # 减法赋值

# 部分更新
w[0, 0].assign(10.0)                     # 更新单个元素
```

## 5. 数据类型（dtype）

TensorFlow 支持多种数据类型：

```python
# 浮点数
tf.float16  # 半精度（省内存，但精度低）
tf.float32  # 单精度（默认，性能与精度平衡）
tf.float64  # 双精度（高精度科学计算）

# 整数
tf.int32    # 32位整数（默认）
tf.int64    # 64位整数

# 布尔值
tf.bool     # True/False

# 字符串
tf.string   # 文本数据
```

**类型转换**：

```python
x = tf.constant([1, 2, 3], dtype=tf.int32)
y = tf.cast(x, dtype=tf.float32)  # 转换为浮点数
```

**混合精度训练**（提速 + 省显存）：

```python
from tensorflow.keras import mixed_precision

# 启用混合精度
mixed_precision.set_global_policy('mixed_float16')

# 模型自动使用 float16 计算，float32 存储
```

## 6. 常见错误与调试技巧

### 错误1：形状不匹配

```python
a = tf.constant([[1, 2]])      # shape: (1, 2)
b = tf.constant([[3], [4]])    # shape: (2, 1)

# c = a + b  # 错误！形状不兼容

# 解决方案1：广播
c = a + tf.transpose(b)  # (1, 2) + (1, 2) ✓

# 解决方案2：重塑
c = tf.reshape(a, [2, 1]) + b  # (2, 1) + (2, 1) ✓
```

### 错误2：类型不匹配

```python
a = tf.constant([1, 2, 3], dtype=tf.int32)
b = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

# c = a + b  # 错误！类型不匹配

# 解决方案：显式转换
c = tf.cast(a, tf.float32) + b  # ✓
```

### 调试技巧

```python
# 1. 打印张量值
tensor = tf.constant([1, 2, 3])
print(tensor.numpy())  # 转为 NumPy 打印

# 2. 打印形状和类型
tf.print("Shape:", tensor.shape, "Dtype:", tensor.dtype)

# 3. 在 @tf.function 中调试
@tf.function
def debug_function(x):
    tf.print("x =", x)  # 使用 tf.print，不是 print
    return x ** 2

# 4. 禁用即时执行（不推荐，仅调试）
# tf.config.run_functions_eagerly(True)
```

## 🔗 下一步

- [02 - Keras API](./02-keras-api.md) - 使用高级 API 快速构建模型
- [03 - 数据管道](./03-data-pipeline.md) - tf.data 高效加载数据
- [04 - 模型训练](./04-model-building.md) - 完整训练流程

## 📚 参考资源

- [TensorFlow 官方教程](https://www.tensorflow.org/tutorials)
- [tf.GradientTape 文档](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
- [Eager Execution 指南](https://www.tensorflow.org/guide/eager)
