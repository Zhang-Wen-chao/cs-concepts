# 原子操作

> 无锁的线程安全操作

## 核心概念

**原子操作 = 不可分割的操作，要么全做，要么不做**

```cpp
// ❌ 非原子：counter++ 分三步
int counter = 0;
counter++;  // 1. 读取  2. 加1  3. 写回

// ✅ 原子：一步完成
std::atomic<int> counter(0);
counter++;  // 原子操作，不会被打断
```

## 为什么需要原子操作？

**问题：锁的开销**
```cpp
// 用锁：每次都要加锁/解锁
std::mutex mtx;
int counter = 0;

for (int i = 0; i < 1000000; ++i) {
    std::lock_guard<std::mutex> lock(mtx);  // 加锁开销
    counter++;
}
```

**解决：原子操作（更快）**
```cpp
// 用原子操作：无锁，硬件直接支持
std::atomic<int> counter(0);

for (int i = 0; i < 1000000; ++i) {
    counter++;  // 快！
}
```

## 基本用法

### 声明和初始化

```cpp
#include <atomic>

std::atomic<int> a(0);       // 初始化为 0
std::atomic<bool> flag(false);
std::atomic<int*> ptr(nullptr);

// 读取
int value = a.load();
int value2 = a;  // 隐式调用 load()

// 写入
a.store(10);
a = 10;  // 隐式调用 store()
```

### 常用操作

```cpp
std::atomic<int> counter(0);

// 1. 自增/自减
counter++;     // fetch_add(1)
counter--;     // fetch_sub(1)
++counter;
--counter;

// 2. 加/减
counter += 5;  // fetch_add(5)
counter -= 3;  // fetch_sub(3)

// 3. 读取-修改-写入（原子）
int old = counter.fetch_add(1);   // 返回旧值，然后 +1
int old2 = counter.fetch_sub(1);  // 返回旧值，然后 -1

// 4. 交换
int old_value = counter.exchange(100);  // 设为 100，返回旧值

// 5. 比较并交换（CAS）
int expected = 10;
int desired = 20;
bool success = counter.compare_exchange_strong(expected, desired);
// 如果 counter == expected，设为 desired，返回 true
// 否则，expected 被设为 counter 的当前值，返回 false
```

## 原子 bool（标志位）

```cpp
std::atomic<bool> flag(false);

// 设置
flag.store(true);
flag = true;

// 读取
bool value = flag.load();
bool value2 = flag;

// test_and_set：设为 true，返回旧值
bool old = flag.test_and_set();  // 原子地设为 true

// clear：设为 false
flag.clear();
```

**经典用法：自旋锁**
```cpp
class SpinLock {
    std::atomic<bool> flag_{false};
public:
    void lock() {
        while (flag_.exchange(true)) {
            // 自旋等待，直到成功获取锁
        }
    }

    void unlock() {
        flag_.store(false);
    }
};
```

## compare_exchange（CAS）

**最强大的原子操作：比较并交换**

```cpp
std::atomic<int> counter(0);

int expected = 0;
int desired = 1;

// strong 版本（推荐）
if (counter.compare_exchange_strong(expected, desired)) {
    // 成功：counter 从 0 变成 1
} else {
    // 失败：expected 被更新为 counter 的当前值
}
```

**用途：无锁数据结构**
```cpp
// 无锁栈的 push
void push(int value) {
    Node* new_node = new Node(value);
    new_node->next = head.load();

    // CAS 循环，直到成功
    while (!head.compare_exchange_weak(new_node->next, new_node)) {
        // 失败：head 被其他线程修改，重试
    }
}
```

**weak vs strong**：
- `compare_exchange_weak`：可能虚假失败（性能更好）
- `compare_exchange_strong`：不会虚假失败（更可靠）

**使用建议**：
- 简单场景 → `strong`
- 循环中 → `weak`（性能更好）

## 内存顺序（高级）

**默认：最强的顺序保证**
```cpp
std::atomic<int> a(0);
a.store(1);  // 默认：memory_order_seq_cst（顺序一致性）
```

**六种内存顺序**：
```cpp
memory_order_relaxed    // 最弱：只保证原子性
memory_order_consume    // 很少用
memory_order_acquire    // 读操作
memory_order_release    // 写操作
memory_order_acq_rel    // 读-修改-写
memory_order_seq_cst    // 最强：顺序一致性（默认）
```

**常用组合**：
```cpp
// 生产者-消费者
std::atomic<bool> ready(false);
int data = 0;

// 生产者
void producer() {
    data = 42;
    ready.store(true, std::memory_order_release);  // 写操作
}

// 消费者
void consumer() {
    while (!ready.load(std::memory_order_acquire)) {}  // 读操作
    std::cout << data;  // 保证看到 42
}
```

**性能排序**：
```
relaxed（最快） > acquire/release > seq_cst（最慢但最安全）
```

**建议**：
- 初学者：用默认（`memory_order_seq_cst`）
- 性能关键：考虑 `acquire/release`
- 简单计数器：`relaxed` 可能够用

## 原子操作 vs 锁

### 原子操作（适用场景）

```cpp
// ✅ 简单计数器
std::atomic<int> counter(0);
counter++;

// ✅ 标志位
std::atomic<bool> done(false);
done = true;

// ✅ 简单的读-改-写
std::atomic<int> value(0);
int old = value.exchange(10);
```

**优点**：
- 更快（无锁）
- 无死锁
- 适合简单操作

**缺点**：
- 只能用于简单类型
- 不能保护多个变量
- 复杂操作难写

### 锁（适用场景）

```cpp
// ✅ 保护多个变量
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);
    data1 = 10;
    data2 = 20;
    data3 = 30;
}

// ✅ 复杂操作
{
    std::lock_guard<std::mutex> lock(mtx);
    if (balance >= amount) {
        balance -= amount;
        process_payment();
    }
}
```

**优点**：
- 能保护任意复杂操作
- 代码清晰易懂

**缺点**：
- 更慢（加锁开销）
- 可能死锁

## 选择建议

```cpp
简单计数器/标志位     → atomic
保护多个变量          → mutex + lock_guard
复杂操作              → mutex + lock_guard
性能关键的简单操作    → atomic
```

## 常见陷阱

### 陷阱 1：非原子的复合操作

```cpp
// ❌ 不是原子的
std::atomic<int> counter(0);
if (counter == 0) {
    counter = 1;  // 其他线程可能在这之前修改了 counter
}

// ✅ 用 CAS
int expected = 0;
counter.compare_exchange_strong(expected, 1);
```

### 陷阱 2：以为可以保护其他变量

```cpp
// ❌ 错误：ready 是原子的，但 data 不是
std::atomic<bool> ready(false);
int data = 0;  // 不是原子的

// 线程 1
data = 42;  // 数据竞争
ready = true;

// 线程 2
if (ready) {
    std::cout << data;  // 可能读到错误的值
}

// ✅ 正确：用内存顺序
data = 42;
ready.store(true, std::memory_order_release);

while (!ready.load(std::memory_order_acquire)) {}
std::cout << data;  // 保证正确
```

### 陷阱 3：复杂类型

```cpp
// ❌ 编译错误：复杂类型不支持原子操作
struct MyStruct {
    int a, b, c;
};
std::atomic<MyStruct> s;  // 可能不支持

// ✅ 简单类型
std::atomic<int> a(0);
std::atomic<bool> b(false);
std::atomic<int*> ptr(nullptr);
```

## 核心要点

1. **原子操作 = 不可分割的操作**
2. **用于简单类型的无锁同步**（int、bool、指针）
3. **比锁快，但功能有限**
4. **常用操作**：`load`、`store`、`fetch_add`、`exchange`、`compare_exchange`
5. **适用场景**：计数器、标志位、简单状态
6. **复杂操作用锁，简单操作用原子**
7. **内存顺序**：初学者用默认，性能关键再优化
