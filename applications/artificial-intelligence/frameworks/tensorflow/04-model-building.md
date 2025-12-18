# 模型构建与训练进阶

> 自定义训练循环、混合精度训练、模型优化技巧

## 🎯 学习目标

前面我们学习了用 `model.fit()` 进行训练，这对大多数场景够用。但有时你需要更细粒度的控制：
- 自定义损失计算逻辑
- 实现复杂的训练策略（如 GAN、强化学习）
- 调试训练过程
- 优化训练性能

## 1. 自定义训练循环

### 为什么需要自定义训练循环？

`model.fit()` 的局限：
- ❌ 无法实现复杂的损失函数（如多任务学习）
- ❌ 无法精细控制梯度更新（如梯度裁剪、梯度累积）
- ❌ 无法实现对抗训练（GAN）
- ❌ 难以调试中间过程

### 基础版：手动实现训练循环

```python
import tensorflow as tf

# 1. 准备数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 2. 创建数据集
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(128)

# 3. 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 4. 定义损失函数和优化器
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 5. 训练循环
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # 遍历每个批次
    for step, (x_batch, y_batch) in enumerate(train_dataset):

        # 前向传播 + 反向传播
        with tf.GradientTape() as tape:
            # 预测
            logits = model(x_batch, training=True)

            # 计算损失
            loss = loss_fn(y_batch, logits)

        # 计算梯度
        gradients = tape.gradient(loss, model.trainable_variables)

        # 更新权重
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 打印进度
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}")
```

### 完整版：带评估指标的训练循环

```python
import tensorflow as tf

# 创建评估指标
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

# 训练一个 epoch
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 更新指标
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y, logits)

    return loss

# 验证一个 epoch
def val_step(x, y):
    logits = model(x, training=False)
    loss = loss_fn(y, logits)

    val_loss_metric.update_state(loss)
    val_acc_metric.update_state(y, logits)

# 完整训练循环
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # 重置指标
    train_loss_metric.reset_states()
    train_acc_metric.reset_states()
    val_loss_metric.reset_states()
    val_acc_metric.reset_states()

    # 训练
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)

    # 验证
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)

    # 打印结果
    print(f"Loss: {train_loss_metric.result():.4f}, "
          f"Accuracy: {train_acc_metric.result():.4f}")
    print(f"Val Loss: {val_loss_metric.result():.4f}, "
          f"Val Accuracy: {val_acc_metric.result():.4f}")
```

### 使用 @tf.function 加速

```python
# 将训练步骤编译为静态图
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y, logits)

# 同样编译验证步骤
@tf.function
def val_step(x, y):
    logits = model(x, training=False)
    loss = loss_fn(y, logits)

    val_loss_metric.update_state(loss)
    val_acc_metric.update_state(y, logits)

# 训练速度提升 2-3 倍！
```

## 2. 自定义损失函数

### 简单自定义损失

```python
def custom_mse_loss(y_true, y_pred):
    """自定义均方误差"""
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)

# 使用
model.compile(optimizer='adam', loss=custom_mse_loss)
```

### 带权重的损失

```python
def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0):
    """处理类别不平衡：正样本权重更高"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # 正样本权重 × pos_weight，负样本权重 × 1
    weights = y_true * (pos_weight - 1) + 1
    return tf.reduce_mean(bce * weights)

# 使用
loss_fn = lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, pos_weight=3.0)
model.compile(optimizer='adam', loss=loss_fn)
```

### 多任务损失

```python
def multi_task_loss(y_true, y_pred):
    """
    y_true = [classification_labels, regression_targets]
    y_pred = [classification_logits, regression_predictions]
    """
    cls_labels, reg_targets = y_true
    cls_logits, reg_preds = y_pred

    # 分类损失
    cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
        cls_labels, cls_logits, from_logits=True
    )

    # 回归损失
    reg_loss = tf.keras.losses.mean_squared_error(reg_targets, reg_preds)

    # 加权组合
    total_loss = cls_loss + 0.5 * reg_loss
    return total_loss
```

### Focal Loss（处理困难样本）

```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss：让模型更关注困难样本
    论文：https://arxiv.org/abs/1708.02002
    """
    # 二分类交叉熵
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # 预测概率
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    # Focal 权重：(1 - p_t)^gamma
    focal_weight = tf.pow(1 - p_t, gamma)

    # 类别权重
    alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

    return tf.reduce_mean(alpha_weight * focal_weight * bce)
```

## 3. 自定义指标

```python
class F1Score(tf.keras.metrics.Metric):
    """自定义 F1 分数指标"""

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        # F1 = 2 * (precision * recall) / (precision + recall)
        return 2 * p * r / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# 使用
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[F1Score()]
)
```

## 4. 梯度裁剪与累积

### 梯度裁剪（防止梯度爆炸）

```python
@tf.function
def train_step_with_clipping(x, y, clip_value=1.0):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)

    gradients = tape.gradient(loss, model.trainable_variables)

    # 方法1：裁剪梯度值
    clipped_gradients = [
        tf.clip_by_value(grad, -clip_value, clip_value)
        for grad in gradients
    ]

    # 方法2：裁剪梯度范数（更常用）
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
```

### 梯度累积（模拟大 batch size）

```python
# 当显存不足时，用小 batch + 梯度累积模拟大 batch
accumulation_steps = 4  # 累积 4 个 batch 再更新

@tf.function
def train_step_with_accumulation(x, y, accumulation_steps):
    # 累积梯度的变量
    accumulated_gradients = [
        tf.Variable(tf.zeros_like(var), trainable=False)
        for var in model.trainable_variables
    ]

    for step in range(accumulation_steps):
        # 取一小批数据
        x_batch = x[step * batch_size:(step + 1) * batch_size]
        y_batch = y[step * batch_size:(step + 1) * batch_size]

        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss = loss_fn(y_batch, logits) / accumulation_steps  # 除以步数

        # 计算梯度
        gradients = tape.gradient(loss, model.trainable_variables)

        # 累积梯度
        for i, grad in enumerate(gradients):
            accumulated_gradients[i].assign_add(grad)

    # 应用累积的梯度
    optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))

    # 重置累积梯度
    for grad_var in accumulated_gradients:
        grad_var.assign(tf.zeros_like(grad_var))
```

## 5. 混合精度训练

### 什么是混合精度？

- **float32**（单精度）：默认，精度高，速度慢，显存占用大
- **float16**（半精度）：精度低，速度快（Tensor Core 加速），显存占用小
- **混合精度**：计算用 float16，存储用 float32，兼顾速度和精度

### 启用混合精度

```python
from tensorflow.keras import mixed_precision

# 全局启用混合精度
mixed_precision.set_global_policy('mixed_float16')

# 创建模型（自动使用 float16）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 最后一层需要 float32 输出（数值稳定性）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', dtype='mixed_float16'),
    tf.keras.layers.Dense(10, dtype='float32')  # ← 输出层用 float32
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 正常训练（速度提升 2-3 倍）
model.fit(train_dataset, epochs=10)
```

### 自定义训练循环中的混合精度

```python
# 使用 Loss Scale 防止梯度下溢
optimizer = tf.keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)

        # Loss Scaling
        scaled_loss = optimizer.get_scaled_loss(loss)

    # 计算缩放后的梯度
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)

    # 反缩放梯度
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)

    # 应用梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
```

### 混合精度的性能对比

```python
import time

# float32（默认）
mixed_precision.set_global_policy('float32')
model_fp32 = create_model()
start = time.time()
model_fp32.fit(train_dataset, epochs=1)
time_fp32 = time.time() - start

# mixed_float16
mixed_precision.set_global_policy('mixed_float16')
model_fp16 = create_model()
start = time.time()
model_fp16.fit(train_dataset, epochs=1)
time_fp16 = time.time() - start

print(f"FP32: {time_fp32:.2f}s")
print(f"FP16: {time_fp16:.2f}s")
print(f"加速: {time_fp32 / time_fp16:.2f}x")

# 典型结果：
# FP32: 120.5s
# FP16: 45.3s
# 加速: 2.66x
```

## 6. 学习率调度

### 学习率衰减策略

```python
# 1. 指数衰减
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 2. 余弦退火
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=10000
)

# 3. 分段常数衰减
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[1000, 2000],
    values=[0.001, 0.0005, 0.0001]
)

# 4. 多项式衰减
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    end_learning_rate=0.0001,
    power=2.0
)
```

### Warm-up + 余弦衰减

```python
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, initial_lr, target_lr):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr

    def __call__(self, step):
        # Warm-up 阶段：线性增长
        warmup_lr = self.initial_lr * step / self.warmup_steps

        # Cosine 衰减阶段
        decay_steps = self.total_steps - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(
            3.14159 * (step - self.warmup_steps) / decay_steps
        ))
        decayed_lr = (self.initial_lr - self.target_lr) * cosine_decay + self.target_lr

        # 组合
        return tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)

# 使用
lr_schedule = WarmUpCosineDecay(
    warmup_steps=1000,
    total_steps=10000,
    initial_lr=0.001,
    target_lr=0.00001
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### 使用回调函数调整学习率

```python
# ReduceLROnPlateau：验证损失不下降时降低学习率
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,         # 学习率减半
    patience=3,         # 3 个 epoch 不改善
    min_lr=1e-7
)

model.fit(train_dataset, validation_data=val_dataset, callbacks=[reduce_lr])
```

## 7. 早停与模型检查点

```python
# 早停
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 保存最佳模型
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 训练
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, checkpoint]
)
```

## 8. TensorBoard 可视化

```python
# 创建 TensorBoard 回调
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,       # 记录权重分布
    write_graph=True,       # 记录计算图
    update_freq='epoch'     # 每个 epoch 更新一次
)

# 训练
model.fit(train_dataset, callbacks=[tensorboard_callback], epochs=10)

# 启动 TensorBoard
# 在终端运行：tensorboard --logdir=./logs
# 浏览器打开：http://localhost:6006
```

### 自定义 TensorBoard 日志

```python
# 创建文件写入器
train_writer = tf.summary.create_file_writer('logs/train')
val_writer = tf.summary.create_file_writer('logs/val')

for epoch in range(epochs):
    for step, (x, y) in enumerate(train_dataset):
        loss = train_step(x, y)

        # 记录训练损失
        with train_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch * steps_per_epoch + step)

    # 记录验证指标
    with val_writer.as_default():
        tf.summary.scalar('accuracy', val_acc, step=epoch)

        # 记录图像
        tf.summary.image('predictions', images, step=epoch, max_outputs=4)

        # 记录直方图
        for var in model.trainable_variables:
            tf.summary.histogram(var.name, var, step=epoch)
```

## 🔗 下一步

- [实践项目](./practices/) - 动手实现完整的深度学习项目

## 📚 参考资源

- [自定义训练循环](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
- [混合精度训练](https://www.tensorflow.org/guide/mixed_precision)
- [TensorBoard 指南](https://www.tensorflow.org/tensorboard)
