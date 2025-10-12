# Concurrency and Parallelism - 并发与并行

> 如何同时处理多个任务？并发和并行有什么区别？

## 🎯 核心概念

### 并发 (Concurrency)
**同时处理多个任务**，但不一定同时执行

```
想象一个厨师：
- 煮面的同时，等水开的时候切菜
- 炒菜的同时，等锅热的时候洗碗
→ 一个人，多个任务交替进行
```

### 并行 (Parallelism)
**同时执行多个任务**

```
想象多个厨师：
- 厨师A煮面
- 厨师B切菜
- 厨师C洗碗
→ 多个人，真正同时进行
```

---

## 🔍 并发 vs 并行

### 关键区别

| 特性 | 并发 (Concurrency) | 并行 (Parallelism) |
|-----|-------------------|-------------------|
| **定义** | 处理多个任务 | 执行多个任务 |
| **核心** | CPU时间片轮转 | 多个CPU同时工作 |
| **硬件** | 单核也可以 | 需要多核 |
| **目的** | 提高响应性 | 提高吞吐量 |
| **例子** | 浏览器多标签页 | 视频渲染 |

### 形象比喻

```
并发 = 一个人在多个任务间快速切换
🧑 → 任务A → 任务B → 任务A → 任务C → 任务B
     (看起来像同时进行，实际是快速切换)

并行 = 多个人同时各做各的
🧑 → 任务A
👩 → 任务B  } 真正同时
🧔 → 任务C
```

### 经典名言

> "并发是关于结构，并行是关于执行"
> "Concurrency is about dealing with lots of things at once.
> Parallelism is about doing lots of things at once."
> — Rob Pike (Go语言设计者)

---

## 🧵 线程与进程

### 进程 (Process)
**操作系统资源分配的基本单位**

```python
# Python - 创建进程
from multiprocessing import Process

def worker(name):
    print(f"进程 {name} 开始工作")

# 创建两个独立的进程
p1 = Process(target=worker, args=("A",))
p2 = Process(target=worker, args=("B",))

p1.start()
p2.start()
```

**特点**：
- 🏠 独立的内存空间
- 🔒 相互隔离，安全
- 🐌 创建和切换开销大
- 💾 通信成本高

### 线程 (Thread)
**进程内的执行单元**

```python
# Python - 创建线程
from threading import Thread

def worker(name):
    print(f"线程 {name} 开始工作")

# 创建两个线程
t1 = Thread(target=worker, args=("A",))
t2 = Thread(target=worker, args=("B",))

t1.start()
t2.start()
```

**特点**：
- 🏡 共享进程的内存空间
- ⚡ 轻量，创建快
- ⚠️ 需要同步，容易出错
- 💬 通信方便

### 对比

```
进程关系：
[进程1: 内存空间1]  [进程2: 内存空间2]
   独立运行            独立运行

线程关系：
[进程: 共享内存空间]
  ├─ 线程1
  ├─ 线程2
  └─ 线程3
    (共享数据，需要同步)
```

---

## 🔐 同步原语

### 1. 锁 (Lock/Mutex)
**互斥锁，同时只能一个线程访问**

```python
from threading import Lock

balance = 0
lock = Lock()

def deposit(amount):
    global balance
    lock.acquire()  # 获取锁
    try:
        temp = balance
        temp += amount
        balance = temp
    finally:
        lock.release()  # 释放锁

# 更优雅的写法
def withdraw(amount):
    with lock:  # 自动获取和释放
        balance -= amount
```

**问题**：死锁 (Deadlock)

```python
lock1 = Lock()
lock2 = Lock()

# 线程A
with lock1:
    with lock2:  # 等待lock2
        # 操作...

# 线程B
with lock2:
    with lock1:  # 等待lock1
        # 操作...

# 结果：A等B释放lock2，B等A释放lock1 → 死锁！
```

### 2. 信号量 (Semaphore)
**限制同时访问的线程数量**

```python
from threading import Semaphore

# 最多3个线程同时访问
semaphore = Semaphore(3)

def access_resource():
    with semaphore:
        print(f"访问资源")
        # 做一些操作...
        # 如果已有3个线程在用，第4个会等待
```

**应用场景**：
- 连接池（限制数据库连接数）
- 限流（API请求限制）

### 3. 条件变量 (Condition Variable)
**线程间的信号机制**

```python
from threading import Condition

condition = Condition()
items = []

# 生产者
def producer():
    with condition:
        items.append("item")
        condition.notify()  # 通知等待的消费者

# 消费者
def consumer():
    with condition:
        while not items:
            condition.wait()  # 等待通知
        item = items.pop()
        print(f"消费: {item}")
```

### 4. 事件 (Event)
**简单的信号标志**

```python
from threading import Event

event = Event()

# 等待线程
def waiter():
    print("等待事件...")
    event.wait()  # 阻塞，直到事件被设置
    print("事件发生！")

# 触发线程
def trigger():
    print("触发事件")
    event.set()  # 设置事件，唤醒所有等待的线程
```

---

## 🔄 常见并发模型

### 1. 多线程模型

```python
from concurrent.futures import ThreadPoolExecutor

def task(n):
    return n * n

# 线程池：复用线程，避免频繁创建
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(task, range(10))
    print(list(results))
```

**优点**：
- ✅ 轻量级
- ✅ 共享内存方便

**缺点**：
- ❌ Python有GIL（全局解释器锁），多线程不能真正并行
- ❌ 竞态条件、死锁风险

### 2. 多进程模型

```python
from concurrent.futures import ProcessPoolExecutor

def cpu_intensive_task(n):
    # CPU密集型任务
    return sum(i * i for i in range(n))

# 进程池
with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(cpu_intensive_task, [1000000] * 4)
    print(list(results))
```

**优点**：
- ✅ 真正的并行（绕过GIL）
- ✅ 隔离性好

**缺点**：
- ❌ 开销大
- ❌ 通信复杂

### 3. 异步编程 (Async/Await)

```python
import asyncio

async def fetch_data(url):
    print(f"开始获取 {url}")
    await asyncio.sleep(2)  # 模拟网络请求
    print(f"完成获取 {url}")
    return f"数据来自 {url}"

async def main():
    # 并发执行多个异步任务
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    print(results)

asyncio.run(main())
```

**特点**：
- 🎯 单线程并发
- 🔄 事件循环驱动
- 💡 适合I/O密集型任务
- 📉 避免线程切换开销

**工作原理**：
```
事件循环：
1. 开始任务1 → 遇到await → 暂停
2. 开始任务2 → 遇到await → 暂停
3. 开始任务3 → 遇到await → 暂停
4. 任务1完成 → 继续
5. 任务2完成 → 继续
6. 任务3完成 → 继续
```

### 4. Actor模型

```python
# 概念示例（伪代码）
class Actor:
    def __init__(self):
        self.mailbox = Queue()

    def send_message(self, message):
        self.mailbox.put(message)

    def process_messages(self):
        while True:
            msg = self.mailbox.get()
            self.handle(msg)
```

**特点**：
- 📬 通过消息传递通信
- 🔒 每个Actor独立状态
- 🌐 易于分布式

**语言**：Erlang, Akka (Scala/Java)

---

## ⚠️ 常见问题

### 1. 竞态条件 (Race Condition)

```python
# 问题代码
counter = 0

def increment():
    global counter
    temp = counter  # 读取
    temp += 1       # 计算
    counter = temp  # 写回

# 两个线程同时执行
# 线程1: 读取0 → 计算1 → 写回1
# 线程2: 读取0 → 计算1 → 写回1
# 结果：counter = 1 (应该是2！)
```

**解决**：使用锁

```python
lock = Lock()

def increment():
    with lock:
        global counter
        counter += 1
```

### 2. 死锁 (Deadlock)

**条件**（必须同时满足）：
1. 互斥：资源不能共享
2. 持有并等待：持有资源同时等待其他资源
3. 不可抢占：资源不能被强制释放
4. 循环等待：A等B，B等A

**预防策略**：
- 按顺序获取锁
- 使用超时
- 避免嵌套锁

### 3. 活锁 (Livelock)

```
两个人在走廊相遇：
A向左 → B向左
A向右 → B向右
A向左 → B向左
...
都在动，但都过不去
```

### 4. 饥饿 (Starvation)

```
线程A一直获取不到资源，因为：
- 优先级太低
- 其他线程占用时间太长
```

---

## 🎯 选择指南

### I/O密集型任务
**推荐**：异步编程 > 多线程

```python
# 网络请求、文件读写、数据库查询
async def fetch_many_urls():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### CPU密集型任务
**推荐**：多进程

```python
# 科学计算、图像处理、数据分析
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(heavy_computation, data_chunks)
```

### 混合型任务
**推荐**：多进程 + 异步

```python
# 每个进程运行异步事件循环
def run_async_in_process():
    asyncio.run(async_tasks())

with ProcessPoolExecutor() as executor:
    executor.map(run_async_in_process, range(4))
```

---

## 💡 最佳实践

### 1. 尽量避免共享状态

```python
# ❌ 共享状态
shared_data = []
lock = Lock()

def worker():
    with lock:
        shared_data.append(...)

# ✅ 消息传递
from queue import Queue

queue = Queue()

def worker():
    queue.put(...)
```

### 2. 使用高级抽象

```python
# ❌ 手动管理线程
threads = []
for i in range(10):
    t = Thread(target=task)
    t.start()
    threads.append(t)
for t in threads:
    t.join()

# ✅ 使用线程池
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(task, range(10))
```

### 3. 不变性优先

```python
# ✅ 使用不可变数据结构
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
# 线程安全，无需锁
```

### 4. 正确关闭资源

```python
executor = ThreadPoolExecutor()
try:
    results = executor.map(task, data)
finally:
    executor.shutdown(wait=True)  # 等待所有任务完成
```

---

## 🔗 相关概念

- [内存管理](memory-management.md) - 多线程的内存可见性
- [编程范式](programming-paradigms.md) - 函数式编程避免共享状态
- [操作系统](../../systems/operating-systems/) - 进程调度、线程实现
- [分布式系统](../../systems/) - 跨机器的并发

---

## 📚 深入学习

- **书籍**：《Java并发编程实战》、《Seven Concurrency Models in Seven Weeks》
- **语言**：Go（goroutine）、Erlang（Actor）、Rust（无畏并发）
- **工具**：async/await、RxJS、Akka

---

**记住**：
1. 并发是结构，并行是执行
2. 能避免共享就避免共享
3. I/O用异步，CPU用多进程
4. 优先使用高级抽象，避免手动管理
5. 并发很难，能不用就不用！
