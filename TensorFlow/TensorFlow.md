## TFRecord
TFRecord 是 TensorFlow 推荐的一种用于存储和读取大量数据的、高效的、标准的二进制文件格式。
把它想象成一个特制的“数据午餐盒”，专门为 TensorFlow 的“胃口”设计，可以快速、有序地提供大量“食物”（数据）。
为什么需要 TFRecord？
当你在训练一个深度学习模型时，尤其是在处理大型数据集（如几百万张图片）时，数据读取本身可能成为一个巨大的性能瓶颈。
想象一下，如果你的数据集包含 100 万张独立的 JPEG 图片文件。在每个训练周期（epoch）中，你的程序都需要：
从硬盘上找到这 100 万个文件中的每一个。
打开文件。
读取内容。
关闭文件。
这个过程涉及大量的硬盘寻道和文件 I/O 操作，速度非常慢。
TFRecord 的解决方案是：将这些零散的小文件打包成一个（或几个）大的、结构化的二进制文件。这样，程序就可以从硬盘上进行一次性的、连续的、流式读取，大大提高了数据加载的效率。
## tensorflow_io
arrow module: Arrow Dataset.

audio module: tensorflow_io.audio

bigquery module: Cloud BigQuery Client for TensorFlow.

bigtable module: tensorflow_io.bigtable

experimental module: tensorflow_io.experimental

genome module: Genomics related ops for Tensorflow.

image module: tensorflow_io.image

version module: tensorflow_io.version
# 自定义
NumPy 数组与 `tf.Tensor` 之间最明显的区别是：

1. 张量可以驻留在加速器内存（例如 GPU、TPU）中。
2. 张量不可变。
# 分布式训练
## 多工作器（worker）配置
现在让我们进入多工作器(worker) 训练的世界。在 TensorFlow 中，需要 TF_CONFIG 环境变量来训练多台机器，每台机器可能具有不同的角色。 TF_CONFIG用于指定作为集群一部分的每个 worker 的集群配置。

TF_CONFIG 有两个组件：cluster 和 task 。 cluster 提供有关训练集群的信息，这是一个由不同类型的工作组成的字典，例如 worker 。在多工作器（worker）培训中，除了常规的“工作器”之外，通常还有一个“工人”承担更多责任，比如保存检查点和为 TensorBoard 编写摘要文件。这样的工作器（worker）被称为“主要”工作者，习惯上worker 中 index 0被指定为主要的 worker（事实上这就是tf.distribute.Strategy的实现方式）。 另一方面，task 提供当前任务的信息。

多工作器是什么？ 就是把原来一台电脑干的活，分给多台电脑一起干。
为什么要这么做？ 快！ 人多力量大，原来要算一年的模型，现在可能几周甚至几天就算完了。
核心思想是什么？
复制模型： 每个工作器都有一份一模一样的模型副本。
瓜分数据： 每个工作器处理一小部分数据。
并行计算： 大家同时在自己的数据上进行计算，得出如何“改进模型”的建议（梯度）。
聚合更新： 把所有人的“建议”汇总起来，得出一个最佳更新方案，然后用这个方案去更新所有人的模型副本，确保大家版本一致。
重复以上步骤。
## 参数服务器训练
参数服务器训练是一种常见的数据并行方法，用于在多台机器上扩展模型训练。

参数服务器训练集群由工作进程和参数服务器组成。变量在参数服务器上创建，并在每个步骤中由工作进程读取和更新。默认情况下，工作进程会独立读取和更新这些变量，而不会彼此同步。因此，参数服务器式训练有时也称为异步训练。