# Operating Systems - 操作系统

> 管理计算机硬件和软件资源的核心系统软件

## 🎯 什么是操作系统？

**操作系统 (OS)** 是管理计算机硬件和软件资源的系统软件，为应用程序提供服务。

### 类比理解

**操作系统就像一个公司的总经理**：
- **管理资源** - 分配CPU、内存、硬盘等硬件资源
- **协调工作** - 让多个程序同时运行互不干扰
- **提供服务** - 为应用程序提供文件管理、网络通信等功能
- **保护安全** - 防止程序互相干扰和恶意攻击

---

## 🏗️ 操作系统的核心功能

### 1. 进程管理 (Process Management)
- 创建、调度、终止进程
- 进程间通信
- 进程同步与互斥
- 死锁处理

### 2. 内存管理 (Memory Management)
- 内存分配与回收
- 虚拟内存
- 分页与分段
- 内存保护

### 3. 文件系统 (File System)
- 文件的组织、存储、检索
- 目录管理
- 文件权限与安全
- 磁盘空间管理

### 4. I/O管理 (I/O Management)
- 设备驱动程序
- 缓冲与缓存
- 中断处理
- 磁盘调度

### 5. 网络管理
- 网络协议栈
- Socket通信
- 网络资源管理

---

## 📚 学习路径

### 核心概念（推荐顺序）

1. **[进程与线程](processes-threads.md)** ⭐⭐⭐
   - 进程的概念
   - 线程的概念
   - 进程调度算法
   - 上下文切换

2. **[进程同步与互斥](synchronization.md)** ⭐⭐⭐
   - 临界区问题
   - 互斥锁、信号量
   - 死锁的概念和处理
   - 经典同步问题

3. **[内存管理](memory-management.md)** ⭐⭐⭐
   - 物理内存与虚拟内存
   - 分页与分段
   - 页面置换算法
   - 内存分配策略

4. **[文件系统](file-systems.md)** ⭐⭐
   - 文件的组织方式
   - 目录结构
   - 磁盘管理
   - 文件系统类型

5. **[I/O系统](io-systems.md)** ⭐⭐
   - I/O设备分类
   - I/O控制方式
   - 缓冲技术
   - 磁盘调度

---

## 🔑 核心概念速览

### 进程 vs 线程

```
进程 (Process)
- 资源分配的基本单位
- 拥有独立的地址空间
- 进程间通信开销大
- 例子：打开一个浏览器

线程 (Thread)
- CPU调度的基本单位
- 共享进程的地址空间
- 线程间通信开销小
- 例子：浏览器中的多个标签页
```

### 进程状态

```
        创建
         ↓
    [新建 New]
         ↓
    [就绪 Ready] ←→ [运行 Running]
         ↑              ↓
         └─── [阻塞 Blocked]
                   ↓
              [终止 Terminated]
```

### 调度算法对比

| 算法 | 特点 | 优点 | 缺点 |
|-----|------|------|------|
| **FCFS** | 先来先服务 | 简单 | 短作业等待长 |
| **SJF** | 短作业优先 | 平均等待短 | 长作业饥饿 |
| **RR** | 时间片轮转 | 响应快 | 上下文切换多 |
| **优先级** | 按优先级 | 灵活 | 可能饥饿 |

### 内存管理技术

```
物理内存不够用？
    ↓
虚拟内存技术
    ↓
┌──────────────────┐
│  程序看到的地址   │ (虚拟地址)
│  0x00001000      │
└──────────────────┘
        ↓ MMU转换
┌──────────────────┐
│  实际物理地址     │ (物理地址)
│  0x12345000      │
└──────────────────┘
        ↓
可以映射到磁盘 (Swap)
```

---

## 🖥️ 常见操作系统

### 类Unix系统
- **Linux** - 开源、服务器主流
- **macOS** - 基于Unix、用户友好
- **BSD** - 学术、网络

### Windows系列
- **Windows** - 桌面主流、企业应用

### 移动系统
- **Android** - 基于Linux
- **iOS** - 基于Unix

### 实时系统
- **RTOS** - 嵌入式、工控

---

## 💡 重要原理

### 1. 时间片轮转
```python
# 模拟时间片轮转调度
from collections import deque

def round_robin(processes, time_quantum):
    """
    processes: [(pid, burst_time), ...]
    time_quantum: 时间片大小
    """
    queue = deque(processes)
    time = 0

    while queue:
        pid, burst = queue.popleft()

        if burst > time_quantum:
            # 执行一个时间片
            print(f"Time {time}: Process {pid} runs for {time_quantum}ms")
            time += time_quantum
            queue.append((pid, burst - time_quantum))
        else:
            # 执行完毕
            print(f"Time {time}: Process {pid} runs for {burst}ms (finished)")
            time += burst

# 使用
processes = [('P1', 10), ('P2', 5), ('P3', 8)]
round_robin(processes, time_quantum=4)
```

### 2. 生产者-消费者问题
```python
from threading import Semaphore, Thread
from queue import Queue

buffer = Queue(maxsize=5)
empty = Semaphore(5)  # 空位数量
full = Semaphore(0)   # 产品数量

def producer():
    for i in range(10):
        empty.acquire()  # 等待空位
        buffer.put(i)
        print(f"Produced {i}")
        full.release()   # 增加产品数

def consumer():
    for i in range(10):
        full.acquire()   # 等待产品
        item = buffer.get()
        print(f"Consumed {item}")
        empty.release()  # 增加空位数

# 启动生产者和消费者
Thread(target=producer).start()
Thread(target=consumer).start()
```

### 3. 页面置换算法
```python
def fifo_page_replacement(pages, frame_count):
    """FIFO页面置换算法"""
    frames = []
    page_faults = 0

    for page in pages:
        if page not in frames:
            page_faults += 1
            if len(frames) < frame_count:
                frames.append(page)
            else:
                frames.pop(0)  # 移除最早的
                frames.append(page)
            print(f"Page fault! Frames: {frames}")
        else:
            print(f"Page hit! Frames: {frames}")

    return page_faults

# 使用
pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
faults = fifo_page_replacement(pages, frame_count=3)
print(f"Total page faults: {faults}")
```

---

## 🔗 相关概念

- [并发与并行](../../fundamentals/programming-concepts/concurrency-parallelism.md) - 多线程编程
- [内存管理](../../fundamentals/programming-concepts/memory-management.md) - 内存基础
- [进程与线程](processes-threads.md) - 详细讲解

---

## 📖 学习建议

### 入门路径
1. 先理解进程和线程的概念
2. 学习进程调度算法
3. 理解内存管理（虚拟内存）
4. 学习进程同步（锁、信号量）
5. 了解文件系统基础

### 进阶学习
- 阅读操作系统经典书籍（如《操作系统概念》）
- 动手实践：写多线程程序
- 研究Linux内核源码
- 做操作系统实验（如MIT 6.828）

### 实践建议
```bash
# Linux下的实用命令

# 查看进程
ps aux
top / htop

# 查看内存
free -h
vmstat

# 查看文件系统
df -h
mount

# 查看I/O
iostat
lsof
```

---

**记住**：
1. 操作系统是硬件和应用之间的桥梁
2. 核心功能：进程管理、内存管理、文件系统、I/O管理
3. 重点理解：进程/线程、调度、虚拟内存、同步
4. 实践很重要：写多线程程序，使用Linux命令
5. 操作系统的设计充满权衡（trade-offs）

**开始学习操作系统的核心概念！** 🚀
