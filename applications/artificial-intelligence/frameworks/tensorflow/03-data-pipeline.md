# tf.data - 高效数据管道

> 构建高性能、可扩展的数据输入流水线

## 🎯 为什么需要 tf.data？

### 传统方法的问题

```python
# ❌ 低效的数据加载方式
import numpy as np

# 一次性加载所有数据到内存
x_train = np.load('train_images.npy')  # 可能几个GB！
y_train = np.load('train_labels.npy')

model.fit(x_train, y_train, epochs=10)
```

**问题**：
1. 内存不足：大数据集无法全部加载
2. GPU 空闲：数据加载时 GPU 在等待
3. 无法扩展：无法处理分布式数据

### tf.data 的优势

```python
# ✅ 高效的数据管道
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

model.fit(dataset, epochs=10)
```

**优势**：
- ⚡ **流式加载**：按需加载，不占用大量内存
- ⚡ **预取（Prefetch）**：CPU 和 GPU 并行工作
- ⚡ **并行处理**：多线程加速数据预处理
- 📈 **可扩展**：支持分布式训练

## 1. 创建 Dataset

### 方式 1：从内存数据创建

```python
import tensorflow as tf
import numpy as np

# 从 NumPy 数组创建
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])

dataset = tf.data.Dataset.from_tensor_slices((x, y))

# 遍历数据
for x_item, y_item in dataset:
    print(f"x: {x_item.numpy()}, y: {y_item.numpy()}")

# 输出：
# x: 1, y: 10
# x: 2, y: 20
# ...
```

### 方式 2：从文件路径创建

```python
# 图像文件列表
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
labels = [0, 1, 0]

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # 归一化
    return image, label

dataset = dataset.map(load_image)
```

### 方式 3：从生成器创建

```python
def data_generator():
    for i in range(100):
        # 模拟从数据库/API 获取数据
        x = np.random.rand(28, 28, 1)
        y = np.random.randint(0, 10)
        yield x, y

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
```

### 方式 4：从 TFRecord 文件创建

```python
# TFRecord 是 TensorFlow 的高效二进制格式
filenames = ['data_part1.tfrecord', 'data_part2.tfrecord']
dataset = tf.data.TFRecordDataset(filenames)

def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    return image, parsed['label']

dataset = dataset.map(parse_tfrecord)
```

## 2. 数据转换操作

### map() - 数据预处理

```python
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

# 对每个元素应用函数
dataset = dataset.map(lambda x: x * 2)
# 结果：[2, 4, 6, 8, 10]

# 图像预处理示例
def preprocess_image(image, label):
    # 数据增强
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)

    # 归一化
    image = (image - 127.5) / 127.5  # [-1, 1]

    return image, label

dataset = dataset.map(preprocess_image)
```

### batch() - 批次处理

```python
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7])

# 创建批次
dataset = dataset.batch(3)

for batch in dataset:
    print(batch.numpy())

# 输出：
# [1 2 3]
# [4 5 6]
# [7]  ← 最后一个批次可能不完整

# 丢弃不完整批次
dataset = dataset.batch(3, drop_remainder=True)
```

### shuffle() - 打乱数据

```python
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

# buffer_size：随机缓冲区大小
dataset = dataset.shuffle(buffer_size=5, seed=42)

# buffer_size 的含义：
# - 太小：打乱不够随机
# - 太大：占用内存多
# - 推荐：数据集大小（小数据集）或 10000+（大数据集）
```

### repeat() - 重复数据集

```python
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])

# 重复 3 次
dataset = dataset.repeat(3)  # [1,2,3,1,2,3,1,2,3]

# 无限重复（常用于训练）
dataset = dataset.repeat()
```

### filter() - 过滤数据

```python
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# 只保留偶数
dataset = dataset.filter(lambda x: x % 2 == 0)  # [2, 4, 6]
```

### take() / skip() - 截取数据

```python
dataset = tf.data.Dataset.from_tensor_slices(range(10))

train_dataset = dataset.skip(2).take(6)  # [2, 3, 4, 5, 6, 7]
test_dataset = dataset.take(2)            # [0, 1]
```

## 3. 性能优化

### prefetch() - 预取数据（最重要！）

```python
# ❌ 没有预取：GPU 等待 CPU 加载数据
dataset = dataset.batch(32)

# ✅ 使用预取：CPU 提前准备下一批数据
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
#                                     ↑ 自动调优缓冲区大小
```

**工作原理**：

```
没有 prefetch:
CPU: [加载batch1] 空闲      [加载batch2] 空闲      ...
GPU: 空闲        [训练batch1] 空闲        [训练batch2] ...

使用 prefetch:
CPU: [加载batch1] [加载batch2] [加载batch3] ...
GPU: 空闲        [训练batch1] [训练batch2] ...
     ↑ GPU训练时，CPU同时准备下一批数据
```

### cache() - 缓存数据

```python
# 第一次迭代后，数据缓存在内存中
dataset = dataset.cache()

# 缓存到磁盘（数据量大时）
dataset = dataset.cache('/tmp/my_cache')

# 典型用法：缓存 → 打乱 → 批次 → 预取
dataset = (dataset
    .cache()               # 1. 缓存原始数据
    .shuffle(10000)        # 2. 打乱
    .batch(32)             # 3. 批次
    .prefetch(tf.data.AUTOTUNE)  # 4. 预取
)
```

### map() 并行化

```python
# 使用多线程并行处理
dataset = dataset.map(
    preprocess_function,
    num_parallel_calls=tf.data.AUTOTUNE  # 自动调优线程数
)
```

### interleave() - 并行读取多个文件

```python
# 从多个文件并行读取
files = tf.data.Dataset.list_files('data/*.tfrecord')

dataset = files.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=4,        # 同时读取 4 个文件
    num_parallel_calls=tf.data.AUTOTUNE
)
```

## 4. 完整的优化流程

### 标准模板（推荐）

```python
def create_dataset(file_pattern, batch_size, is_training=True):
    """创建高性能数据管道"""

    # 1. 加载文件列表
    files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    # 2. 并行读取文件
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 3. 解析数据
    dataset = dataset.map(
        parse_example,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 4. 缓存（如果数据集不大）
    if is_training:
        dataset = dataset.cache()

    # 5. 打乱（仅训练时）
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    # 6. 批次处理
    dataset = dataset.batch(batch_size)

    # 7. 数据增强（仅训练时）
    if is_training:
        dataset = dataset.map(
            augment_data,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 8. 重复（训练时无限循环）
    if is_training:
        dataset = dataset.repeat()

    # 9. 预取（最重要！）
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# 使用
train_dataset = create_dataset('train/*.tfrecord', batch_size=32, is_training=True)
val_dataset = create_dataset('val/*.tfrecord', batch_size=64, is_training=False)

model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```

### 操作顺序很重要！

```python
# ✅ 正确顺序：先缓存再打乱
dataset = (dataset
    .map(parse_function)    # 解析
    .cache()                # 缓存解析后的数据
    .shuffle(10000)         # 打乱（每个epoch重新打乱）
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# ❌ 错误顺序：先打乱再缓存
dataset = (dataset
    .map(parse_function)
    .shuffle(10000)         # 打乱
    .cache()                # 缓存打乱后的数据（每个epoch顺序相同！）
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
```

## 5. 实战示例

### 示例 1：MNIST 数据管道

```python
import tensorflow as tf

# 加载 MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 创建训练集
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_dataset = (train_dataset
    .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))  # 归一化
    .cache()                    # 缓存
    .shuffle(10000)             # 打乱
    .batch(128)                 # 批次
    .prefetch(tf.data.AUTOTUNE) # 预取
)

# 创建测试集
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

test_dataset = (test_dataset
    .map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

# 训练
model.fit(train_dataset, validation_data=test_dataset, epochs=10)
```

### 示例 2：图像分类数据管道

```python
import tensorflow as tf
import pathlib

# 图像目录结构：
# data/
#   ├── train/
#   │   ├── cat/
#   │   │   ├── img1.jpg
#   │   │   └── img2.jpg
#   │   └── dog/
#   │       ├── img1.jpg
#   │       └── img2.jpg

data_dir = pathlib.Path('data/train')

# 使用 Keras 工具创建数据集
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

# 自定义预处理
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # 数据增强
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

train_dataset = (train_dataset
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
```

### 示例 3：文本数据管道

```python
import tensorflow as tf

# 文本数据
texts = ['I love TensorFlow', 'Deep learning is awesome', ...]
labels = [1, 1, 0, ...]

# 创建 TextVectorization 层
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=50
)

# 适配词汇表
vectorize_layer.adapt(texts)

# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices((texts, labels))

dataset = (dataset
    .map(lambda text, label: (vectorize_layer(text), label))
    .cache()
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
```

### 示例 4：处理大型数据集

```python
# 假设有 1TB 图像数据，无法全部加载到内存

def process_large_dataset(file_pattern):
    # 1. 创建文件列表
    files = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.Dataset.from_tensor_slices(files)

    # 2. 并行读取文件
    def load_and_preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0

        # 从文件名提取标签
        label = tf.strings.split(path, '/')[-2]
        label = tf.cast(label == 'cat', tf.int32)

        return image, label

    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 3. 优化流水线
    dataset = (dataset
        .shuffle(10000)             # 大缓冲区
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset

# 使用
train_dataset = process_large_dataset('data/train/*/*.jpg')
model.fit(train_dataset, epochs=10, steps_per_epoch=1000)
```

## 6. 调试与检查

### 查看数据集内容

```python
# 取出前几个样本查看
for image, label in train_dataset.take(3):
    print(f"Image shape: {image.shape}, Label: {label.numpy()}")

# 可视化批次
import matplotlib.pyplot as plt

for images, labels in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()
```

### 性能分析

```python
import time

# 测试数据管道性能
dataset = create_dataset(...)

# 预热
for _ in dataset.take(10):
    pass

# 测试
start = time.time()
for i, batch in enumerate(dataset.take(100)):
    if i % 10 == 0:
        print(f"Batch {i}: {time.time() - start:.2f}s")

# 使用 TensorFlow Profiler
tf.profiler.experimental.start('logs')
model.fit(dataset, epochs=1)
tf.profiler.experimental.stop()
```

### 常见问题排查

```python
# 问题1：内存不足
# 解决：减小 batch_size 或 buffer_size
dataset = dataset.batch(16)  # 减小批次
dataset = dataset.shuffle(1000)  # 减小缓冲区

# 问题2：数据加载太慢
# 解决：增加并行度
dataset = dataset.map(fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# 问题3：训练速度慢
# 解决：使用 cache() 缓存数据
dataset = dataset.cache()

# 问题4：数据不随机
# 解决：增大 shuffle buffer_size
dataset = dataset.shuffle(10000)  # 至少是 batch_size 的几倍
```

## 7. 高级技巧

### 自定义数据增强

```python
@tf.function  # 编译为静态图加速
def augment(image, label):
    # 随机裁剪
    image = tf.image.random_crop(image, size=[224, 224, 3])

    # 随机翻转
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # 随机亮度/对比度
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # 随机色相/饱和度
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_saturation(image, 0.8, 1.2)

    # 归一化
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
```

### 混合精度训练

```python
# 启用混合精度
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# 数据类型转换
def cast_to_fp16(image, label):
    image = tf.cast(image, tf.float16)
    return image, label

dataset = dataset.map(cast_to_fp16)
```

### 分布式训练

```python
# 多 GPU 训练
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # 创建数据集（自动分片）
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)

    # 创建模型
    model = create_model()
    model.compile(...)

# 训练
model.fit(train_dataset, epochs=10)
```

## 🔗 下一步

- [04 - 模型构建与训练](./04-model-building.md) - 自定义训练循环、混合精度训练
- [实践项目](./practices/) - 动手实现完整项目

## 📚 参考资源

- [tf.data 官方指南](https://www.tensorflow.org/guide/data)
- [tf.data 性能优化](https://www.tensorflow.org/guide/data_performance)
- [数据增强最佳实践](https://www.tensorflow.org/tutorials/images/data_augmentation)
