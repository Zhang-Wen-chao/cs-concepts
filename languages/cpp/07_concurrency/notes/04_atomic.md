# std::atomic、load/store、fetch_add、CAS、memory_order

## 1. 为什么需要原子操作

没有原子操作时，多个线程读写同一变量是**数据竞争**（UB）。用 mutex 太重，原子操作是**无锁**的轻量级方案：

```cpp
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> counter{0};  // 原子变量

void increment() {
    for (int i = 0; i < 100000; ++i) {
        counter.fetch_add(1);  // 原子加
    }
}

int main() {
    std::thread t1(increment), t2(increment);
    t1.join(); t2.join();
    std::cout << counter << '\n';  // 200000（保证正确）
}
```

---

## 2. 基本操作

```cpp
#include <atomic>

std::atomic<int> a{0};

// 读
int v1 = a.load();       // 原子读
int v2 = a;              // 等价于 load

// 写
a.store(42);             // 原子写
a = 42;                  // 等价于 store

// 读改写（RMW）
int old = a.fetch_add(1);   // a += 1，返回旧值
int old = a.fetch_sub(1);   // a -= 1
int old = a.exchange(99);   // 写入 99，返回旧值
```

**原子类型**：`std::atomic<bool>`、`std::atomic<int>`、`std::atomic<long>`、`std::atomic<size_t>` 等，以及 `std::atomic<T*>`。

**`std::atomic_flag`**：最小的原子布尔型，保证无锁，用于自旋锁。

---

## 3. CAS — Compare-And-Swap

CAS 是无锁编程的基石：如果当前值等于期望值，就更新，否则不做任何事。

```cpp
#include <atomic>

std::atomic<int> val{10};

void cas_example() {
    int expected = 10;
    bool ok = val.compare_exchange_weak(expected, 100);
    // 如果 val == 10，则 val = 100，返回 true
    // 否则 expected 被更新为 val 当前值，返回 false
}

// 实战：无锁 push 到链表
struct Node { int data; Node* next; };
std::atomic<Node*> head{nullptr};

void push(int x) {
    Node* new_node = new Node{x, nullptr};
    new_node->next = head.load();
    while (!head.compare_exchange_weak(new_node->next, new_node)) {
        // CAS 失败时 new_node->next 已被更新为最新 head
    }
}
```

**`compare_exchange_weak` vs `compare_exchange_strong`**：weak 可能在 spurious fail 时返回 false（更高效用于循环），strong 保证不虚假失败。

---

## 4. memory_order — 内存序

控制原子操作之间的可见性顺序：

```cpp
#include <atomic>
#include <thread>

std::atomic<bool> ready{false};
int data = 0;

// 生产者
void producer() {
    data = 42;                              // 普通写
    ready.store(true, std::memory_order_release);  // release：之前的写对 consumer 可见
}

// 消费者
void consumer() {
    while (!ready.load(std::memory_order_acquire)) {}  // acquire：看到 producer 的写
    // data == 42 有保证！
}
```

| memory_order | 含义 |
|-------------|------|
| `relaxed` | 只保证原子性，不保证顺序 |
| `consume` | 数据依赖序（基本不用） |
| `acquire` | 后面的读看到 release 前的写 |
| `release` | 前面的写对 acquire 可见 |
| `acq_rel` | acquire + release（用于 RMW） |
| `seq_cst` | 全局顺序一致（默认，最慢但最安全） |

**无锁编程很难**：绝大多数场景用默认 `seq_cst` 就够了，不需要手动调优 memory_order。

---

## 5. 自旋锁（用 atomic_flag 实现）

```cpp
#include <atomic>

class SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            // 忙等
        }
    }
    void unlock() {
        flag.clear(std::memory_order_release);
    }
};
```

---

## 总结

| 操作 | 用途 |
|------|------|
| `load` / `store` | 原子读/写 |
| `fetch_add` / `fetch_sub` | 原子加减 |
| `exchange` | 原子交换 |
| `compare_exchange_weak/strong` | CAS，无锁编程基础 |
| memory_order | 控制可见性（默认 seq_cst） |
