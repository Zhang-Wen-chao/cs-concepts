# Process Synchronization - 进程同步与互斥

> 多个进程/线程如何安全地访问共享资源？如何避免数据竞争？

## 🎯 为什么需要同步？

### 问题：数据竞争 (Race Condition)

```python
# 两个线程同时执行这段代码
counter = 0  # 共享变量

def increment():
    global counter
    for _ in range(100000):
        counter += 1  # 这不是原子操作！

# 运行
import threading
t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(counter)  # 期望: 200000, 实际: 可能小于200000！
```

**为什么会出错？**

```
counter += 1 实际上是三步：

线程1: 读取 counter (值=0)
线程2: 读取 counter (值=0)
线程1: 计算 0+1=1
线程2: 计算 0+1=1
线程1: 写回 1
线程2: 写回 1

结果：counter=1，而不是2！
```

---

## 🔑 临界区 (Critical Section)

**临界区**：访问共享资源的代码段

### 临界区问题的要求

1. **互斥 (Mutual Exclusion)**：同一时刻只有一个进程在临界区
2. **进步 (Progress)**：如果没有进程在临界区，想进入的进程应该能进入
3. **有限等待 (Bounded Waiting)**：进程等待进入临界区的时间是有限的

### 临界区结构

```python
while True:
    # 进入区 (Entry Section)
    acquire_lock()

    # 临界区 (Critical Section)
    # 访问共享资源
    counter += 1

    # 退出区 (Exit Section)
    release_lock()

    # 剩余区 (Remainder Section)
    # 其他代码
```

---

## 🔒 互斥锁 (Mutex)

### 基本使用

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        lock.acquire()  # 获取锁
        counter += 1
        lock.release()  # 释放锁

# 或者使用with语句（更安全）
def increment_safe():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

# 运行
t1 = threading.Thread(target=increment_safe)
t2 = threading.Thread(target=increment_safe)
t1.start()
t2.start()
t1.join()
t2.join()

print(counter)  # 200000 ✓
```

### 锁的实现原理

```python
class SimpleLock:
    def __init__(self):
        self.locked = False

    def acquire(self):
        """获取锁"""
        while self.locked:  # 忙等待
            pass  # 自旋
        self.locked = True

    def release(self):
        """释放锁"""
        self.locked = False
```

**问题**：这个实现不是原子的！需要硬件支持。

### 硬件支持：Test-and-Set

```python
def test_and_set(target):
    """原子操作：测试并设置"""
    old_value = target
    target = True
    return old_value

class SpinLock:
    def __init__(self):
        self.lock = False

    def acquire(self):
        while test_and_set(self.lock):
            pass  # 自旋等待

    def release(self):
        self.lock = False
```

---

## 🚦 信号量 (Semaphore)

**信号量**：一个整数变量，支持两个原子操作：

- **P (wait/down)**: 信号量减1，如果小于0则阻塞
- **V (signal/up)**: 信号量加1，唤醒一个等待的进程

### 二元信号量（互斥锁）

```python
import threading

# 二元信号量 = 互斥锁
mutex = threading.Semaphore(1)

counter = 0

def increment():
    global counter
    for _ in range(100000):
        mutex.acquire()  # P操作
        counter += 1
        mutex.release()  # V操作
```

### 计数信号量（资源池）

```python
# 最多允许3个线程同时访问
semaphore = threading.Semaphore(3)

def access_resource(thread_id):
    print(f"Thread {thread_id} waiting...")
    semaphore.acquire()
    try:
        print(f"Thread {thread_id} accessing resource")
        time.sleep(2)  # 模拟使用资源
    finally:
        print(f"Thread {thread_id} releasing resource")
        semaphore.release()

# 创建10个线程，但最多3个同时运行
threads = []
for i in range(10):
    t = threading.Thread(target=access_resource, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

### 信号量的实现

```python
class Semaphore:
    def __init__(self, value=1):
        self.value = value
        self.waiting_list = []

    def wait(self):  # P操作
        self.value -= 1
        if self.value < 0:
            # 阻塞当前进程
            self.waiting_list.append(current_process())
            block()

    def signal(self):  # V操作
        self.value += 1
        if self.value <= 0:
            # 唤醒一个等待的进程
            process = self.waiting_list.pop(0)
            wakeup(process)
```

---

## 🍽️ 经典同步问题

### 1. 生产者-消费者问题

**问题描述**：
- 生产者生产数据放入缓冲区
- 消费者从缓冲区取数据
- 缓冲区大小有限

```python
import threading
import time
from queue import Queue

# 使用Queue（线程安全）
buffer = Queue(maxsize=5)

def producer(producer_id):
    for i in range(10):
        item = f"Item-{producer_id}-{i}"
        buffer.put(item)  # 自动阻塞当缓冲区满
        print(f"Producer {producer_id} produced {item}")
        time.sleep(0.1)

def consumer(consumer_id):
    while True:
        item = buffer.get()  # 自动阻塞当缓冲区空
        if item is None:  # 结束信号
            break
        print(f"Consumer {consumer_id} consumed {item}")
        time.sleep(0.2)

# 运行
producers = [threading.Thread(target=producer, args=(i,)) for i in range(2)]
consumers = [threading.Thread(target=consumer, args=(i,)) for i in range(2)]

for p in producers:
    p.start()
for c in consumers:
    c.start()

for p in producers:
    p.join()

# 发送结束信号
for _ in consumers:
    buffer.put(None)

for c in consumers:
    c.join()
```

**用信号量实现**：

```python
import threading
from collections import deque

buffer = deque(maxlen=5)
empty = threading.Semaphore(5)  # 空位数量
full = threading.Semaphore(0)   # 产品数量
mutex = threading.Lock()        # 保护buffer

def producer(producer_id):
    for i in range(10):
        item = f"Item-{producer_id}-{i}"

        empty.acquire()  # 等待空位
        with mutex:
            buffer.append(item)
            print(f"Producer {producer_id} produced {item}")
        full.release()  # 增加产品数

        time.sleep(0.1)

def consumer(consumer_id):
    for i in range(10):
        full.acquire()  # 等待产品
        with mutex:
            item = buffer.popleft()
            print(f"Consumer {consumer_id} consumed {item}")
        empty.release()  # 增加空位数

        time.sleep(0.2)
```

---

### 2. 读者-写者问题

**问题描述**：
- 多个读者可以同时读
- 写者独占访问（不能有其他读者或写者）

```python
import threading
import time

class ReadWriteLock:
    def __init__(self):
        self.readers = 0
        self.writer = False
        self.mutex = threading.Lock()
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()

    def acquire_read(self):
        """获取读锁"""
        with self.mutex:
            self.readers += 1
            if self.readers == 1:
                self.write_lock.acquire()  # 第一个读者阻止写者

    def release_read(self):
        """释放读锁"""
        with self.mutex:
            self.readers -= 1
            if self.readers == 0:
                self.write_lock.release()  # 最后一个读者允许写者

    def acquire_write(self):
        """获取写锁"""
        self.write_lock.acquire()

    def release_write(self):
        """释放写锁"""
        self.write_lock.release()

# 使用
rw_lock = ReadWriteLock()
shared_data = 0

def reader(reader_id):
    for _ in range(5):
        rw_lock.acquire_read()
        print(f"Reader {reader_id} reading: {shared_data}")
        time.sleep(0.1)
        rw_lock.release_read()

def writer(writer_id):
    global shared_data
    for i in range(5):
        rw_lock.acquire_write()
        shared_data += 1
        print(f"Writer {writer_id} writing: {shared_data}")
        time.sleep(0.2)
        rw_lock.release_write()

# 运行
readers = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
writers = [threading.Thread(target=writer, args=(i,)) for i in range(2)]

for t in readers + writers:
    t.start()
for t in readers + writers:
    t.join()
```

---

### 3. 哲学家就餐问题

**问题描述**：
- 5个哲学家围坐圆桌
- 每人面前一个盘子，每两人之间一支筷子
- 哲学家要拿起左右两支筷子才能吃饭
- 如何避免死锁？

```python
import threading
import time

NUM_PHILOSOPHERS = 5
forks = [threading.Lock() for _ in range(NUM_PHILOSOPHERS)]

def philosopher(phil_id):
    """哲学家行为"""
    left_fork = phil_id
    right_fork = (phil_id + 1) % NUM_PHILOSOPHERS

    for _ in range(5):
        # 思考
        print(f"Philosopher {phil_id} is thinking...")
        time.sleep(0.1)

        # 拿筷子吃饭
        print(f"Philosopher {phil_id} is hungry")

        # 解决方案1：按顺序拿筷子（避免环路）
        first = min(left_fork, right_fork)
        second = max(left_fork, right_fork)

        with forks[first]:
            with forks[second]:
                print(f"Philosopher {phil_id} is eating")
                time.sleep(0.2)

        print(f"Philosopher {phil_id} finished eating")

# 运行
philosophers = [threading.Thread(target=philosopher, args=(i,))
                for i in range(NUM_PHILOSOPHERS)]

for p in philosophers:
    p.start()
for p in philosophers:
    p.join()
```

**死锁的解决方案**：

1. **按顺序获取资源**：总是先拿编号小的筷子
2. **限制就餐人数**：最多4个哲学家同时拿筷子
3. **奇偶策略**：奇数号先拿左筷子，偶数号先拿右筷子
4. **服务员策略**：需要服务员允许才能拿筷子

```python
# 方案2：限制就餐人数
room = threading.Semaphore(NUM_PHILOSOPHERS - 1)

def philosopher_v2(phil_id):
    left_fork = phil_id
    right_fork = (phil_id + 1) % NUM_PHILOSOPHERS

    for _ in range(5):
        print(f"Philosopher {phil_id} thinking")
        time.sleep(0.1)

        room.acquire()  # 进入房间
        with forks[left_fork]:
            with forks[right_fork]:
                print(f"Philosopher {phil_id} eating")
                time.sleep(0.2)
        room.release()  # 离开房间
```

---

## ☠️ 死锁 (Deadlock)

### 什么是死锁？

**死锁**：多个进程互相等待对方持有的资源，导致都无法继续执行。

```python
import threading

lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1():
    with lock1:
        print("Thread 1 acquired lock1")
        time.sleep(0.1)
        with lock2:  # 等待lock2
            print("Thread 1 acquired lock2")

def thread2():
    with lock2:
        print("Thread 2 acquired lock2")
        time.sleep(0.1)
        with lock1:  # 等待lock1
            print("Thread 2 acquired lock1")

# 可能死锁！
t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)
t1.start()
t2.start()
```

### 死锁的四个必要条件

1. **互斥 (Mutual Exclusion)**：资源不能共享
2. **持有并等待 (Hold and Wait)**：进程持有资源并等待其他资源
3. **非抢占 (No Preemption)**：资源不能被强制夺走
4. **循环等待 (Circular Wait)**：存在进程等待环路

```
P1 持有 R1，等待 R2
P2 持有 R2，等待 R3
P3 持有 R3，等待 R1
       ↓
   形成环路！
```

### 死锁的处理

#### 1. 预防死锁（破坏四个条件之一）

```python
# 破坏循环等待：按顺序获取锁
def thread_safe():
    locks = [lock1, lock2]
    locks.sort(key=id)  # 按内存地址排序

    with locks[0]:
        with locks[1]:
            # 临界区
            pass
```

#### 2. 避免死锁（银行家算法）

```python
class BankersAlgorithm:
    def __init__(self, available, maximum, allocation):
        self.available = available    # 可用资源
        self.maximum = maximum        # 最大需求
        self.allocation = allocation  # 已分配
        self.need = maximum - allocation  # 还需要

    def is_safe(self):
        """检查是否处于安全状态"""
        work = self.available.copy()
        finish = [False] * len(self.allocation)

        while True:
            found = False
            for i in range(len(self.allocation)):
                if not finish[i] and all(self.need[i] <= work):
                    # 可以完成进程i
                    work += self.allocation[i]
                    finish[i] = True
                    found = True

            if not found:
                break

        return all(finish)
```

#### 3. 检测和恢复

```python
class DeadlockDetector:
    def __init__(self):
        self.wait_for_graph = {}  # 等待图

    def add_edge(self, from_process, to_process):
        """添加等待关系"""
        if from_process not in self.wait_for_graph:
            self.wait_for_graph[from_process] = []
        self.wait_for_graph[from_process].append(to_process)

    def has_cycle(self):
        """检测环路（DFS）"""
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.wait_for_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True  # 找到环路

            rec_stack.remove(node)
            return False

        for node in self.wait_for_graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False
```

#### 4. 使用超时机制

```python
import threading

lock1 = threading.Lock()
lock2 = threading.Lock()

def safe_thread():
    while True:
        # 尝试获取锁，设置超时
        if lock1.acquire(timeout=1):
            try:
                if lock2.acquire(timeout=1):
                    try:
                        # 临界区
                        print("Got both locks")
                        break
                    finally:
                        lock2.release()
                else:
                    # 获取lock2超时，释放lock1重试
                    print("Failed to get lock2, retrying...")
            finally:
                lock1.release()
        time.sleep(0.1)  # 随机等待避免活锁
```

---

## 🔐 其他同步机制

### 1. 条件变量 (Condition Variable)

```python
import threading

condition = threading.Condition()
items = []

def producer():
    for i in range(10):
        with condition:
            items.append(i)
            print(f"Produced {i}")
            condition.notify()  # 唤醒等待的消费者
        time.sleep(0.1)

def consumer():
    while True:
        with condition:
            while not items:
                condition.wait()  # 等待生产者
            item = items.pop(0)
            print(f"Consumed {item}")
            if item == 9:
                break

p = threading.Thread(target=producer)
c = threading.Thread(target=consumer)
p.start()
c.start()
p.join()
c.join()
```

### 2. 事件 (Event)

```python
import threading

event = threading.Event()

def waiter():
    print("Waiting for event...")
    event.wait()  # 阻塞直到事件被设置
    print("Event received!")

def setter():
    time.sleep(2)
    print("Setting event")
    event.set()  # 设置事件，唤醒所有等待者

w = threading.Thread(target=waiter)
s = threading.Thread(target=setter)
w.start()
s.start()
w.join()
s.join()
```

### 3. 屏障 (Barrier)

```python
import threading

barrier = threading.Barrier(3)  # 需要3个线程到达

def worker(worker_id):
    print(f"Worker {worker_id} working...")
    time.sleep(worker_id)  # 不同工作时间
    print(f"Worker {worker_id} waiting at barrier")

    barrier.wait()  # 等待其他线程

    print(f"Worker {worker_id} passed barrier")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## 💡 最佳实践

### 1. 避免嵌套锁

```python
# ❌ 危险：嵌套锁容易死锁
with lock1:
    with lock2:
        pass

# ✅ 安全：按顺序获取
locks = sorted([lock1, lock2], key=id)
with locks[0]:
    with locks[1]:
        pass
```

### 2. 使用上下文管理器

```python
# ❌ 可能忘记释放锁
lock.acquire()
try:
    # 临界区
    pass
finally:
    lock.release()

# ✅ 自动释放
with lock:
    # 临界区
    pass
```

### 3. 尽量缩小临界区

```python
# ❌ 临界区太大
with lock:
    data = fetch_data()  # 慢操作
    result = process(data)  # 慢操作
    update_shared(result)

# ✅ 只保护必要部分
data = fetch_data()
result = process(data)
with lock:
    update_shared(result)
```

### 4. 使用线程安全的数据结构

```python
from queue import Queue
from collections import deque

# ✅ 使用内置的线程安全结构
queue = Queue()  # 线程安全

# 而不是
# my_list = []
# lock = threading.Lock()
```

---

## 🔓 无锁编程 (Lock-Free Programming)

### 为什么需要无锁？

锁的问题：
- ⏱️ 上下文切换开销大
- 🔒 可能死锁
- 🐌 线程优先级反转
- 📉 高竞争下性能差

**无锁编程**：使用**原子操作**代替锁，避免线程阻塞。

---

### 原子操作的核心：CAS

**CAS (Compare-And-Swap)** - 比较并交换：

```cpp
// 伪代码
bool CAS(int* memory, int expected, int new_value) {
    // 这是一个原子操作（硬件支持）
    if (*memory == expected) {
        *memory = new_value;
        return true;  // 成功
    }
    return false;  // 失败，值已被其他线程修改
}
```

### C++中的原子操作

```cpp
#include <atomic>
#include <thread>
#include <iostream>

std::atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 100000; ++i) {
        counter.fetch_add(1);  // 原子增加
        // 或者：counter++（对于atomic类型也是原子的）
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);
    t1.join();
    t2.join();

    std::cout << "Counter: " << counter << std::endl;  // 200000 ✓
    return 0;
}
```

### 用CAS实现无锁栈

```cpp
#include <atomic>

template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(const T& d) : data(d), next(nullptr) {}
    };

    std::atomic<Node*> head{nullptr};

public:
    void push(const T& data) {
        Node* new_node = new Node(data);
        // 自旋直到成功
        while (true) {
            Node* old_head = head.load();
            new_node->next = old_head;

            // CAS: 如果head还是old_head，则更新为new_node
            if (head.compare_exchange_weak(old_head, new_node)) {
                return;  // 成功
            }
            // 失败则重试
        }
    }

    bool pop(T& result) {
        while (true) {
            Node* old_head = head.load();
            if (old_head == nullptr) {
                return false;  // 栈空
            }

            Node* new_head = old_head->next;

            // CAS: 尝试移动head
            if (head.compare_exchange_weak(old_head, new_head)) {
                result = old_head->data;
                delete old_head;
                return true;  // 成功
            }
            // 失败则重试
        }
    }
};
```

**工作原理**：
```
线程1: 读取 head=A
线程2: 读取 head=A
线程1: CAS(A, B) → 成功！head=B
线程2: CAS(A, C) → 失败！head已经是B了
线程2: 重试：读取 head=B
线程2: CAS(B, C) → 成功！head=C
```

---

### ABA问题

**问题**：CAS无法区分"没变"和"变了又变回来"

```
时刻0: head = A
时刻1: 线程1读取 head=A，准备CAS
时刻2: 线程2: pop A, pop B, push A  (head又是A)
时刻3: 线程1: CAS(head, A, ...) 成功！但这个A已经不是原来的A了
```

**解决方案**：加版本号

```cpp
template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(const T& d) : data(d), next(nullptr) {}
    };

    struct TaggedPointer {
        Node* ptr;
        size_t tag;  // 版本号
    };

    std::atomic<TaggedPointer> head{{nullptr, 0}};

public:
    void push(const T& data) {
        Node* new_node = new Node(data);
        TaggedPointer old_head, new_head;

        while (true) {
            old_head = head.load();
            new_node->next = old_head.ptr;
            new_head.ptr = new_node;
            new_head.tag = old_head.tag + 1;  // 版本号递增

            if (head.compare_exchange_weak(old_head, new_head)) {
                return;
            }
        }
    }
};
```

---

### 无锁队列

```cpp
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        T data;
        std::atomic<Node*> next{nullptr};
        Node() = default;
        Node(const T& d) : data(d) {}
    };

    std::atomic<Node*> head;
    std::atomic<Node*> tail;

public:
    LockFreeQueue() {
        Node* dummy = new Node();
        head.store(dummy);
        tail.store(dummy);
    }

    void enqueue(const T& data) {
        Node* new_node = new Node(data);

        while (true) {
            Node* last = tail.load();
            Node* next = last->next.load();

            if (last == tail.load()) {  // 确保last还是tail
                if (next == nullptr) {
                    // tail确实指向最后一个节点
                    if (last->next.compare_exchange_weak(next, new_node)) {
                        // 成功添加节点，尝试移动tail
                        tail.compare_exchange_weak(last, new_node);
                        return;
                    }
                } else {
                    // tail落后了，帮助其他线程移动tail
                    tail.compare_exchange_weak(last, next);
                }
            }
        }
    }

    bool dequeue(T& result) {
        while (true) {
            Node* first = head.load();
            Node* last = tail.load();
            Node* next = first->next.load();

            if (first == head.load()) {
                if (first == last) {
                    if (next == nullptr) {
                        return false;  // 队列空
                    }
                    // tail落后，帮助移动
                    tail.compare_exchange_weak(last, next);
                } else {
                    result = next->data;
                    if (head.compare_exchange_weak(first, next)) {
                        delete first;
                        return true;
                    }
                }
            }
        }
    }
};
```

---

### 有锁 vs 无锁对比

| 特性 | 有锁 (Mutex) | 无锁 (Lock-Free) |
|-----|-------------|-----------------|
| **实现难度** | 简单 ⭐ | 复杂 ⭐⭐⭐⭐⭐ |
| **性能（低竞争）** | 好 | 很好 |
| **性能（高竞争）** | 差 | 好 |
| **死锁** | 可能 ❌ | 不会 ✅ |
| **活锁** | 不会 ✅ | 可能 ❌ |
| **上下文切换** | 有 | 无 |
| **CPU使用** | 低 | 高（自旋） |
| **公平性** | 好 | 不保证 |
| **内存回收** | 简单 | 复杂（ABA问题） |

---

### 何时使用无锁？

✅ **适合无锁的场景**：
- 高并发、低竞争环境
- 实时系统（避免阻塞）
- 性能关键路径
- 简单的数据结构（栈、队列）

❌ **不适合无锁的场景**：
- 复杂的临界区逻辑
- 需要多个操作的原子性
- 开发时间紧张
- 维护性要求高

### 性能测试对比

```cpp
// 性能对比示例（伪代码）
// 低竞争（8线程，各做自己的事）:
//   有锁: 100ms
//   无锁: 80ms   (提升20%)

// 高竞争（8线程，抢同一个资源）:
//   有锁: 500ms  (大量阻塞)
//   无锁: 200ms  (提升60%)
```

---

### 无锁编程的层次

```
无阻塞编程的进化：

1. 阻塞 (Blocking)
   - 使用锁，线程可能阻塞
   - 例子：Mutex

2. 无锁 (Lock-Free)
   - 至少有一个线程能在有限步内完成操作
   - 例子：上面的无锁栈、队列

3. 无等待 (Wait-Free)
   - 每个线程都能在有限步内完成操作
   - 更难实现，性能更好
   - 例子：某些特殊的数据结构
```

---

### 实践建议

1. **优先使用锁**
   - 大多数情况下，锁的性能已经足够好
   - 锁更容易理解和维护

2. **谨慎使用无锁**
   - 只在性能瓶颈且经过测试确认时使用
   - 需要深入理解内存模型和原子操作

3. **使用成熟的库**
   ```cpp
   // 使用标准库的无锁结构
   #include <atomic>
   std::atomic<T> atomic_var;

   // 或使用专业的并发库
   // Intel TBB
   // Boost.Lockfree
   // Facebook's Folly
   ```

4. **充分测试**
   - 无锁代码的bug很难复现和调试
   - 使用线程消毒器（Thread Sanitizer）
   - 压力测试

---

## 🔗 相关概念

- [进程与线程](processes-threads.md) - 多线程基础
- [内存管理](memory-management.md) - 共享内存
- [并发与并行](../../fundamentals/programming-concepts/concurrency-parallelism.md) - 并发编程

---

**记住**：
1. 数据竞争是多线程的最大敌人
2. 临界区需要互斥保护
3. 锁、信号量、条件变量各有用途
4. 死锁的四个必要条件
5. 破坏任一条件可预防死锁
6. 尽量缩小临界区范围
7. 使用高级同步原语（条件变量、事件）
8. 嵌套锁要小心，按顺序获取
9. **无锁编程**：用原子操作代替锁，但实现复杂
10. CAS是无锁的核心，但要注意ABA问题
11. 大多数情况优先用锁，特殊场景才考虑无锁
