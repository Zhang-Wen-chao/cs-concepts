# Processes and Threads - 进程与线程

> 操作系统中最核心的概念：程序如何运行？如何并发执行？

## 🎯 什么是进程？

**进程 (Process)** 是程序的一次执行实例，是操作系统进行资源分配和调度的基本单位。

### 生活类比

```
程序 (Program) = 菜谱
进程 (Process) = 根据菜谱做菜的过程

- 菜谱只是一份说明书（静态的代码）
- 做菜的过程是动态的，需要厨师、食材、厨具
- 同一份菜谱可以同时开始多次做菜（多个进程）
```

---

## 📋 进程的组成

一个进程包含：

```
┌─────────────────────────┐
│    进程控制块 (PCB)      │ ← 进程的"身份证"
├─────────────────────────┤
│    代码段 (Text)         │ ← 程序的机器代码
├─────────────────────────┤
│    数据段 (Data)         │ ← 全局变量、静态变量
├─────────────────────────┤
│    堆 (Heap)            │ ← 动态分配的内存
│         ↑               │
│         |               │
│         ↓               │
│    栈 (Stack)           │ ← 函数调用、局部变量
└─────────────────────────┘
```

### 进程控制块 (PCB)

PCB 保存进程的重要信息：

```python
class PCB:
    def __init__(self):
        self.pid = None           # 进程ID
        self.state = None         # 进程状态
        self.program_counter = None  # 程序计数器
        self.registers = {}       # CPU寄存器
        self.memory_limits = {}   # 内存限制
        self.open_files = []      # 打开的文件
        self.priority = None      # 优先级
        self.parent_pid = None    # 父进程ID
```

---

## 🔄 进程的状态

进程在运行过程中会在不同状态之间转换：

```
                创建进程
                   ↓
              ┌────────┐
              │  新建  │
              │  New   │
              └────────┘
                   ↓ 进入就绪队列
              ┌────────┐
         ┌───→│  就绪  │←──┐
         │    │ Ready  │   │ 时间片用完
         │    └────────┘   │ 或被抢占
         │         ↓ 调度  │
         │    ┌────────┐   │
         │    │  运行  │───┘
    I/O  │    │Running │
    完成 │    └────────┘
         │         ↓ 等待I/O
         │    ┌────────┐
         └────│  阻塞  │
              │Blocked │
              └────────┘
                   ↓ 进程结束
              ┌────────┐
              │  终止  │
              │Terminated│
              └────────┘
```

### 状态转换示例

```python
class ProcessState:
    NEW = "新建"
    READY = "就绪"
    RUNNING = "运行"
    BLOCKED = "阻塞"
    TERMINATED = "终止"

class Process:
    def __init__(self, pid):
        self.pid = pid
        self.state = ProcessState.NEW

    def admit(self):
        """进入就绪队列"""
        self.state = ProcessState.READY

    def dispatch(self):
        """被调度运行"""
        self.state = ProcessState.RUNNING

    def wait_for_io(self):
        """等待I/O"""
        self.state = ProcessState.BLOCKED

    def io_complete(self):
        """I/O完成"""
        self.state = ProcessState.READY

    def exit(self):
        """进程结束"""
        self.state = ProcessState.TERMINATED
```

---

## 🚀 进程的创建

### Unix/Linux: fork()

```c
#include <stdio.h>
#include <unistd.h>

int main() {
    pid_t pid = fork();  // 创建子进程

    if (pid < 0) {
        // fork失败
        printf("Fork failed\n");
    } else if (pid == 0) {
        // 子进程
        printf("I am child, PID: %d\n", getpid());
    } else {
        // 父进程
        printf("I am parent, PID: %d, child PID: %d\n", getpid(), pid);
    }

    return 0;
}
```

**fork() 的魔法**：
```
父进程执行fork()
    ↓
创建子进程（完整复制父进程）
    ↓
两个进程从fork()之后继续执行
    ↓
父进程：fork()返回子进程PID
子进程：fork()返回0
```

### Python示例

```python
import os
import time

def child_process():
    print(f"Child process: PID={os.getpid()}, Parent PID={os.getppid()}")
    time.sleep(2)
    print("Child finished")

def parent_process():
    print(f"Parent process: PID={os.getpid()}")

    pid = os.fork()

    if pid == 0:
        # 子进程
        child_process()
    else:
        # 父进程
        print(f"Created child with PID={pid}")
        os.wait()  # 等待子进程结束
        print("Parent finished")

# 在Unix/Linux上运行
parent_process()
```

---

## 🧵 什么是线程？

**线程 (Thread)** 是进程内的一个执行流，是CPU调度的基本单位。

### 进程 vs 线程

```
进程 = 一个公司
├── 线程1 = 员工1
├── 线程2 = 员工2
└── 线程3 = 员工3

- 公司拥有资源（办公室、设备）→ 进程拥有地址空间
- 员工共享公司资源 → 线程共享进程资源
- 员工各自工作 → 线程各自执行
```

### 对比表

| 特性 | 进程 | 线程 |
|-----|------|------|
| **定义** | 资源分配单位 | CPU调度单位 |
| **地址空间** | 独立 | 共享 |
| **资源** | 独立拥有 | 共享进程资源 |
| **通信** | IPC（进程间通信） | 直接读写共享内存 |
| **开销** | 大 | 小 |
| **创建速度** | 慢 | 快 |
| **切换速度** | 慢 | 快 |
| **崩溃影响** | 独立，不影响其他进程 | 影响整个进程 |

---

## 💻 线程的实现

### Python多线程

```python
import threading
import time

# 方法1：继承Thread类
class MyThread(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        for i in range(5):
            print(f"{self.name}: {i}")
            time.sleep(0.5)

# 方法2：传入函数
def worker(name):
    for i in range(5):
        print(f"{name}: {i}")
        time.sleep(0.5)

# 使用
if __name__ == "__main__":
    # 方法1
    t1 = MyThread("Thread-1")
    t1.start()

    # 方法2
    t2 = threading.Thread(target=worker, args=("Thread-2",))
    t2.start()

    # 等待所有线程结束
    t1.join()
    t2.join()

    print("All threads finished")
```

### 线程的生命周期

```
创建线程
    ↓
 start()
    ↓
┌──────┐
│ 就绪 │←──┐
└──────┘   │
    ↓      │
┌──────┐   │
│ 运行 │───┘ 时间片用完
└──────┘
    ↓ run()执行完
┌──────┐
│ 终止 │
└──────┘
```

---

## 🔀 进程调度

### 调度算法

#### 1. 先来先服务 (FCFS - First Come First Served)

```python
def fcfs_scheduling(processes):
    """
    processes: [(pid, arrival_time, burst_time), ...]
    """
    # 按到达时间排序
    processes.sort(key=lambda x: x[1])

    current_time = 0
    waiting_times = []

    for pid, arrival, burst in processes:
        if current_time < arrival:
            current_time = arrival

        waiting_time = current_time - arrival
        waiting_times.append(waiting_time)

        print(f"Process {pid}: Wait {waiting_time}ms, Run at {current_time}ms")
        current_time += burst

    avg_waiting = sum(waiting_times) / len(waiting_times)
    print(f"Average waiting time: {avg_waiting}ms")

# 使用
processes = [
    ('P1', 0, 8),   # (PID, 到达时间, 运行时间)
    ('P2', 1, 4),
    ('P3', 2, 9),
    ('P4', 3, 5)
]
fcfs_scheduling(processes)
```

#### 2. 短作业优先 (SJF - Shortest Job First)

```python
def sjf_scheduling(processes):
    """最短作业优先（非抢占）"""
    processes.sort(key=lambda x: x[1])  # 按到达时间
    ready_queue = []
    current_time = 0
    completed = []

    i = 0
    while i < len(processes) or ready_queue:
        # 将到达的进程加入就绪队列
        while i < len(processes) and processes[i][1] <= current_time:
            ready_queue.append(processes[i])
            i += 1

        if not ready_queue:
            current_time = processes[i][1]
            continue

        # 选择运行时间最短的
        ready_queue.sort(key=lambda x: x[2])
        pid, arrival, burst = ready_queue.pop(0)

        waiting = current_time - arrival
        print(f"Process {pid}: Wait {waiting}ms, Run at {current_time}ms")

        current_time += burst
        completed.append((pid, waiting))

# 使用
processes = [
    ('P1', 0, 8),
    ('P2', 1, 4),
    ('P3', 2, 2),
    ('P4', 3, 1)
]
sjf_scheduling(processes)
```

#### 3. 时间片轮转 (RR - Round Robin)

```python
from collections import deque

def round_robin(processes, time_quantum):
    """
    时间片轮转调度
    processes: [(pid, arrival, burst), ...]
    time_quantum: 时间片大小
    """
    processes.sort(key=lambda x: x[1])
    ready_queue = deque()
    current_time = 0
    remaining = {p[0]: p[2] for p in processes}  # 剩余时间
    i = 0

    # 加入第一个进程
    ready_queue.append(processes[0])
    i = 1

    while ready_queue:
        pid, arrival, burst = ready_queue.popleft()

        # 执行一个时间片
        execute_time = min(time_quantum, remaining[pid])
        current_time += execute_time
        remaining[pid] -= execute_time

        print(f"Time {current_time-execute_time}-{current_time}: Process {pid}")

        # 加入新到达的进程
        while i < len(processes) and processes[i][1] <= current_time:
            ready_queue.append(processes[i])
            i += 1

        # 如果未完成，重新加入队列
        if remaining[pid] > 0:
            ready_queue.append((pid, arrival, burst))

# 使用
processes = [
    ('P1', 0, 10),
    ('P2', 1, 5),
    ('P3', 2, 8)
]
round_robin(processes, time_quantum=4)
```

#### 4. 优先级调度

```python
def priority_scheduling(processes):
    """
    优先级调度（非抢占）
    processes: [(pid, arrival, burst, priority), ...]
    priority越小优先级越高
    """
    processes.sort(key=lambda x: x[1])
    ready_queue = []
    current_time = 0
    i = 0

    while i < len(processes) or ready_queue:
        # 加入到达的进程
        while i < len(processes) and processes[i][1] <= current_time:
            ready_queue.append(processes[i])
            i += 1

        if not ready_queue:
            current_time = processes[i][1]
            continue

        # 选择优先级最高的（数字最小）
        ready_queue.sort(key=lambda x: x[3])
        pid, arrival, burst, priority = ready_queue.pop(0)

        print(f"Process {pid} (Priority {priority}): Run at {current_time}ms")
        current_time += burst

# 使用
processes = [
    ('P1', 0, 10, 3),  # (PID, 到达, 运行, 优先级)
    ('P2', 1, 5, 1),
    ('P3', 2, 8, 2)
]
priority_scheduling(processes)
```

### 调度算法对比

```
FCFS (先来先服务)
✅ 简单
❌ 平均等待时间长
❌ 护航效应（短作业等长作业）

SJF (短作业优先)
✅ 平均等待时间最短
❌ 长作业可能饥饿
❌ 需要预知运行时间

RR (时间片轮转)
✅ 响应时间好
✅ 公平
❌ 上下文切换多
❌ 时间片大小难选择

优先级调度
✅ 灵活
✅ 适合实时系统
❌ 低优先级可能饥饿
```

---

## 🔄 上下文切换

### 什么是上下文切换？

当CPU从一个进程/线程切换到另一个时，需要保存和恢复执行状态。

```
进程A运行
    ↓
保存A的状态到PCB
    ↓
加载B的状态从PCB
    ↓
进程B运行
```

### 上下文包含什么？

```python
class Context:
    def __init__(self):
        # CPU寄存器
        self.program_counter = None  # 程序计数器
        self.stack_pointer = None    # 栈指针
        self.registers = {}          # 通用寄存器

        # 进程状态
        self.state = None

        # 内存管理
        self.page_table = None       # 页表

        # 其他
        self.open_files = []
        self.signal_mask = None
```

### 上下文切换的开销

```
直接开销：
- 保存/恢复寄存器
- 切换地址空间
- 更新内核数据结构

间接开销：
- CPU缓存失效
- TLB失效
- 流水线清空
```

**为什么线程切换比进程快？**
```
进程切换：
- 保存整个进程上下文
- 切换地址空间（页表）
- 刷新TLB缓存

线程切换：
- 只保存线程上下文（寄存器）
- 不需要切换地址空间
- TLB不需要刷新
```

---

## 🔗 进程间通信 (IPC)

### 1. 管道 (Pipe)

```python
import os

# 创建管道
r, w = os.pipe()

pid = os.fork()

if pid == 0:
    # 子进程：写入
    os.close(r)  # 关闭读端
    os.write(w, b"Hello from child!")
    os.close(w)
else:
    # 父进程：读取
    os.close(w)  # 关闭写端
    data = os.read(r, 100)
    print(f"Parent received: {data.decode()}")
    os.close(r)
    os.wait()
```

### 2. 共享内存

```python
from multiprocessing import Process, Value, Array

def worker(shared_value, shared_array):
    shared_value.value = 42
    for i in range(len(shared_array)):
        shared_array[i] *= 2

if __name__ == "__main__":
    # 共享整数
    num = Value('i', 0)
    # 共享数组
    arr = Array('i', [1, 2, 3, 4, 5])

    p = Process(target=worker, args=(num, arr))
    p.start()
    p.join()

    print(f"Shared value: {num.value}")
    print(f"Shared array: {arr[:]}")
```

### 3. 消息队列

```python
from multiprocessing import Process, Queue

def producer(q):
    for i in range(5):
        q.put(f"Message {i}")
        print(f"Produced: Message {i}")

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consumed: {item}")

if __name__ == "__main__":
    q = Queue()

    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    q.put(None)  # 结束信号
    p2.join()
```

### 4. 信号

```python
import signal
import os
import time

def signal_handler(signum, frame):
    print(f"Received signal {signum}")

# 注册信号处理函数
signal.signal(signal.SIGUSR1, signal_handler)

pid = os.fork()

if pid == 0:
    # 子进程
    time.sleep(1)
    os.kill(os.getppid(), signal.SIGUSR1)  # 发送信号给父进程
else:
    # 父进程
    print("Parent waiting for signal...")
    time.sleep(2)
    os.wait()
```

---

## 🆚 多进程 vs 多线程

### 何时用多进程？

✅ **适合场景**：
- CPU密集型任务（利用多核）
- 需要完全隔离（安全性）
- 稳定性要求高（一个崩溃不影响其他）

```python
from multiprocessing import Process, cpu_count

def cpu_intensive(n):
    """CPU密集型任务"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

if __name__ == "__main__":
    processes = []
    for i in range(cpu_count()):
        p = Process(target=cpu_intensive, args=(10000000,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

### 何时用多线程？

✅ **适合场景**：
- I/O密集型任务（等待网络、磁盘）
- 需要频繁通信
- 资源共享

```python
import threading
import time

def io_intensive(url):
    """I/O密集型任务（模拟）"""
    print(f"Fetching {url}...")
    time.sleep(2)  # 模拟网络I/O
    print(f"Finished {url}")

urls = ["url1", "url2", "url3", "url4"]
threads = []

for url in urls:
    t = threading.Thread(target=io_intensive, args=(url,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

---

## 🔗 相关概念

- [并发与并行](../../fundamentals/programming-concepts/concurrency-parallelism.md) - 并发编程
- [进程同步与互斥](synchronization.md) - 线程安全
- [内存管理](memory-management.md) - 进程的内存空间

---

**记住**：
1. 进程是资源分配单位，线程是CPU调度单位
2. 进程独立，线程共享
3. 进程稳定但重，线程轻量但需要同步
4. 调度算法各有优劣，需要权衡
5. 上下文切换有开销
6. CPU密集用多进程，I/O密集用多线程
