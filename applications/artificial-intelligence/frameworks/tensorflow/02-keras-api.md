# Keras API - TensorFlow 的高级接口

> Keras 让深度学习模型构建像搭积木一样简单

## 🎯 Keras 是什么？

Keras 是 TensorFlow 的**高级 API**，提供了简洁、用户友好的接口来构建和训练深度学习模型。

**核心理念**：
- 为人类设计，不是为机器
- 模块化：层、模型、优化器、损失函数都是独立模块
- 易于扩展：可以自定义任何组件

## 1. 三种构建模型的方式

### 方式 1：Sequential API（顺序模型）

**适用场景**：线性堆叠的简单模型（最常用）

```python
import tensorflow as tf
from tensorflow import keras

# 构建模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 或者逐层添加
model = keras.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_shape=(784,)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation='softmax'))
```

**优点**：
- ✅ 代码简洁，易读
- ✅ 适合 90% 的场景

**局限**：
- ❌ 只能处理单输入、单输出
- ❌ 不支持分支或跳跃连接（如 ResNet）

### 方式 2：Functional API（函数式模型）

**适用场景**：复杂架构（多输入、多输出、分支结构）

```python
from tensorflow import keras

# 定义输入
inputs = keras.Input(shape=(784,))

# 定义层之间的连接
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

# 创建模型
model = keras.Model(inputs=inputs, outputs=outputs)
```

**多输入多输出示例**：

```python
# 例子：文本分类 + 情感分析（共享特征提取）
text_input = keras.Input(shape=(100,), name='text')

# 共享的特征提取层
x = keras.layers.Embedding(10000, 128)(text_input)
x = keras.layers.LSTM(64)(x)

# 分支1：文本分类（5个类别）
classification_output = keras.layers.Dense(5, activation='softmax', name='classification')(x)

# 分支2：情感分析（正面/负面）
sentiment_output = keras.layers.Dense(1, activation='sigmoid', name='sentiment')(x)

# 创建多输出模型
model = keras.Model(
    inputs=text_input,
    outputs=[classification_output, sentiment_output]
)
```

**残差连接（ResNet风格）**：

```python
inputs = keras.Input(shape=(32, 32, 3))

# 主路径
x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)

x = keras.layers.Conv2D(64, 3, padding='same')(x)
x = keras.layers.BatchNormalization()(x)

# 跳跃连接（shortcut）
shortcut = keras.layers.Conv2D(64, 1)(inputs)  # 调整通道数
x = keras.layers.Add()([x, shortcut])  # 残差相加
x = keras.layers.Activation('relu')(x)

# 输出
outputs = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation='softmax')(outputs)

model = keras.Model(inputs, outputs)
```

### 方式 3：Model Subclassing（子类化模型）

**适用场景**：需要完全自定义的复杂逻辑（研究、特殊需求）

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:  # 只在训练时应用 Dropout
            x = self.dropout(x)
        return self.dense2(x)

# 创建模型实例
model = MyModel()
```

**高级示例：自定义前向传播**：

```python
class AttentionModel(keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embed_dim)
        self.attention = keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # 自定义前向传播逻辑
        x = self.embedding(inputs)

        # Self-attention
        attention_output = self.attention(query=x, key=x, value=x)

        # 全局平均池化
        x = tf.reduce_mean(attention_output, axis=1)

        return self.dense(x)

model = AttentionModel(vocab_size=10000, embed_dim=128, num_heads=4)
```

### 三种方式对比

| 特性 | Sequential | Functional | Subclassing |
|------|-----------|-----------|-------------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 灵活性 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 调试难度 | 简单 | 中等 | 困难 |
| 模型可视化 | ✅ | ✅ | ❌ |
| 模型保存 | ✅ | ✅ | 部分支持 |
| 适用场景 | 简单线性模型 | 复杂架构 | 研究/特殊需求 |

**推荐选择**：
- 初学者：Sequential
- 工程师：Functional（90%场景够用）
- 研究者：Subclassing

## 2. 常用层（Layers）

### 核心层

```python
from tensorflow.keras import layers

# 1. 全连接层（Dense）
layers.Dense(units=128, activation='relu')

# 2. 卷积层（用于图像）
layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')

# 3. 循环层（用于序列）
layers.LSTM(units=64, return_sequences=True)  # 返回完整序列
layers.GRU(units=64)                          # 轻量级 RNN

# 4. 嵌入层（用于文本/类别特征）
layers.Embedding(input_dim=10000, output_dim=128)

# 5. 归一化层
layers.BatchNormalization()    # 批归一化
layers.LayerNormalization()    # 层归一化（Transformer常用）

# 6. Dropout（防止过拟合）
layers.Dropout(rate=0.5)

# 7. 池化层
layers.MaxPooling2D(pool_size=2)
layers.GlobalAveragePooling2D()  # 全局平均池化
```

### 激活函数

```python
# 作为层
layers.ReLU()
layers.LeakyReLU(alpha=0.2)
layers.Softmax()

# 作为参数
layers.Dense(64, activation='relu')
layers.Dense(10, activation='softmax')

# 常用激活函数对比
# - ReLU: 最常用，速度快，但可能"死亡"
# - LeakyReLU: 解决 ReLU 死亡问题
# - Sigmoid: 输出 [0, 1]，用于二分类
# - Tanh: 输出 [-1, 1]，比 Sigmoid 效果好
# - Softmax: 多分类输出层
```

### 正则化层

```python
# L1/L2 正则化
layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01))

# Dropout（随机丢弃神经元）
layers.Dropout(0.5)

# Spatial Dropout（丢弃整个特征图通道）
layers.SpatialDropout2D(0.2)

# Batch Normalization（批归一化）
layers.BatchNormalization()
```

## 3. 编译模型（Compile）

模型定义后，需要配置训练参数：

```python
model.compile(
    optimizer='adam',                      # 优化器
    loss='sparse_categorical_crossentropy', # 损失函数
    metrics=['accuracy']                    # 评估指标
)
```

### 优化器（Optimizer）

```python
# 1. SGD（随机梯度下降）
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# 2. Adam（自适应学习率，最常用）
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# 3. RMSprop（适合 RNN）
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

# 4. AdamW（Adam + 权重衰减，现代首选）
optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
```

### 损失函数（Loss）

```python
# 回归任务
loss = 'mean_squared_error'        # MSE
loss = 'mean_absolute_error'       # MAE

# 二分类任务
loss = 'binary_crossentropy'       # 标签: [0, 1]

# 多分类任务
loss = 'categorical_crossentropy'  # 标签: one-hot [[1,0,0], [0,1,0]]
loss = 'sparse_categorical_crossentropy'  # 标签: 整数 [0, 1, 2]

# 自定义损失
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model.compile(optimizer='adam', loss=custom_loss)
```

### 评估指标（Metrics）

```python
# 分类任务
metrics = ['accuracy']                          # 准确率
metrics = [keras.metrics.Precision()]           # 精确率
metrics = [keras.metrics.Recall()]              # 召回率
metrics = [keras.metrics.AUC()]                 # AUC

# 回归任务
metrics = [keras.metrics.MeanSquaredError()]    # MSE
metrics = [keras.metrics.MeanAbsoluteError()]   # MAE

# 多个指标
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)
```

## 4. 训练模型（Fit）

```python
# 基本训练
history = model.fit(
    x_train, y_train,          # 训练数据
    batch_size=32,             # 批次大小
    epochs=10,                 # 训练轮数
    validation_data=(x_val, y_val),  # 验证数据
    verbose=1                  # 显示进度条
)

# 使用生成器（大数据集）
history = model.fit(
    train_dataset,             # tf.data.Dataset 对象
    epochs=10,
    validation_data=val_dataset
)
```

### 回调函数（Callbacks）

回调函数在训练过程中执行特定操作：

```python
from tensorflow.keras.callbacks import *

callbacks = [
    # 1. 早停（防止过拟合）
    EarlyStopping(
        monitor='val_loss',        # 监控验证损失
        patience=5,                # 5轮不改善就停止
        restore_best_weights=True  # 恢复最佳权重
    ),

    # 2. 保存最佳模型
    ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),

    # 3. 学习率衰减
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,               # 学习率减半
        patience=3,
        min_lr=1e-7
    ),

    # 4. TensorBoard 可视化
    TensorBoard(log_dir='./logs'),

    # 5. 自定义回调
    LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch}: loss={logs['loss']:.4f}")
    )
]

history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)
```

### 训练历史（History）

```python
import matplotlib.pyplot as plt

# 训练完成后，history 对象包含训练指标
print(history.history.keys())  # ['loss', 'accuracy', 'val_loss', 'val_accuracy']

# 可视化训练过程
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## 5. 评估与预测

### 评估模型

```python
# 在测试集上评估
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 自定义评估
results = model.evaluate(x_test, y_test, return_dict=True)
print(results)  # {'loss': 0.123, 'accuracy': 0.95}
```

### 预测

```python
# 批量预测
predictions = model.predict(x_test)
print(predictions.shape)  # (10000, 10) - 10000个样本，10个类别的概率

# 单样本预测
single_sample = x_test[0:1]  # 保持维度 (1, 784)
prediction = model.predict(single_sample, verbose=0)
predicted_class = np.argmax(prediction)

# 获取类别（不是概率）
predicted_classes = np.argmax(predictions, axis=1)
```

## 6. 模型保存与加载

### 方式 1：SavedModel 格式（推荐）

```python
# 保存整个模型（架构 + 权重 + 优化器状态）
model.save('my_model')  # 创建目录

# 加载模型
loaded_model = keras.models.load_model('my_model')

# 可以直接使用
predictions = loaded_model.predict(x_test)
```

### 方式 2：HDF5 格式

```python
# 保存为 .h5 文件
model.save('my_model.h5')

# 加载
loaded_model = keras.models.load_model('my_model.h5')
```

### 方式 3：只保存权重

```python
# 保存权重
model.save_weights('my_weights.h5')

# 加载权重（需要先定义相同架构）
model = create_model()  # 你的模型定义函数
model.load_weights('my_weights.h5')
```

### 方式 4：保存为 TFLite（移动端部署）

```python
# 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 7. 完整示例：MNIST 手写数字识别

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. 加载数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. 数据预处理
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 3. 构建模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# 4. 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# 6. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 7. 预测
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predicted: {predicted_classes}")
print(f"Actual: {y_test[:5]}")

# 8. 保存模型
model.save('mnist_model.h5')
```

## 8. 实用技巧

### 查看模型结构

```python
model.summary()

# 输出示例：
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 128)               100480
# dropout (Dropout)            (None, 128)               0
# dense_1 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
```

### 可视化模型

```python
keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
```

### 冻结层（迁移学习）

```python
# 加载预训练模型
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False)

# 冻结基础模型的权重
base_model.trainable = False

# 添加自定义顶层
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation='softmax')
])

# 只训练顶层
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

### 获取中间层输出

```python
# 方法1：创建新模型
layer_name = 'dense_1'
intermediate_model = keras.Model(
    inputs=model.input,
    outputs=model.get_layer(layer_name).output
)
intermediate_output = intermediate_model.predict(x_test)

# 方法2：使用 Functional API
from tensorflow.keras import backend as K
get_layer_output = K.function([model.input], [model.layers[2].output])
layer_output = get_layer_output([x_test])[0]
```

## 🔗 下一步

- [03 - 数据管道 tf.data](./03-data-pipeline.md) - 高效加载和预处理数据
- [04 - 模型训练进阶](./04-model-building.md) - 自定义训练循环、混合精度训练
- [实践项目](./practices/) - 动手实现经典模型

## 📚 参考资源

- [Keras 官方文档](https://keras.io/)
- [Sequential API 指南](https://keras.io/guides/sequential_model/)
- [Functional API 指南](https://keras.io/guides/functional_api/)
- [Keras 示例库](https://keras.io/examples/)
