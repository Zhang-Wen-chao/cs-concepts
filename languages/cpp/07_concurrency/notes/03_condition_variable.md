# 条件变量、wait/notify_one/notify_all、生产者消费者
> For a detailed walkthrough with more examples, see `_supplementary/03_condition_variable.md`

## 1. 条件变量解决的问题

线程需要"等待某个条件成立"再继续。轮询（忙等）浪费 CPU，条件变量让线程挂起等待唤醒。

```cpp
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

std::queue<int> q;
std::mutex mtx;
std::condition_variable cv;

void producer() {
    for (int i = 0; i < 10; ++i) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            q.push(i);
            std::cout << "produced " << i << '\n';
        }
        cv.notify_one();                     // 通知一个消费者
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer() {
    for (int i = 0; i < 10; ++i) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, []{ return !q.empty(); });  // 等待条件成立
        int val = q.front();
        q.pop();
        lk.unlock();
        std::cout << "consumed " << val << '\n';
    }
}

int main() {
    std::thread t1(producer), t2(consumer);
    t1.join(); t2.join();
}
```

---

## 2. wait 的两种形式

```cpp
// 形式 1：无条件等待（可能虚假唤醒）
cv.wait(lk);  // 必须用 unique_lock

// 形式 2：带谓词的 wait（推荐，自动处理虚假唤醒）
cv.wait(lk, []{ return condition_is_true(); });
```

**为什么用 `unique_lock`**：`wait()` 需要先解锁 mutex 让其他线程进入，被唤醒后再加锁。`lock_guard` 不支持手动解锁。

---

## 3. 虚假唤醒（Spurious Wakeup）

条件变量可能在没有 `notify` 的情况下自行醒来（硬件原因）。**所以必须用带谓词的 `wait`**，它在醒来后会重新检查条件：

```cpp
// ❌ 错误：可能虚假唤醒，继续执行时 condition 仍未满足
while (q.empty()) {
    cv.wait(lk);
}

// ✅ 正确：谓词版本等价于下面这个循环
cv.wait(lk, []{ return !q.empty(); });
```

---

## 4. notify_one vs notify_all

| 函数 | 效果 | 适用场景 |
|------|------|---------|
| `notify_one()` | 唤醒一个等待线程 | 单个生产者 → 单个消费者 |
| `notify_all()` | 唤醒所有等待线程 | 多读多写/屏障场景 |

---

## 5. 一个更复杂的设计：有界缓冲区

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class BoundedQueue {
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable not_full_, not_empty_;
    size_t capacity_;

public:
    explicit BoundedQueue(size_t cap) : capacity_(cap) {}

    void push(T val) {
        std::unique_lock lk(mtx_);
        not_full_.wait(lk, [this]{ return q_.size() < capacity_; });
        q_.push(std::move(val));
        not_empty_.notify_one();
    }

    T pop() {
        std::unique_lock lk(mtx_);
        not_empty_.wait(lk, [this]{ return !q_.empty(); });
        T val = std::move(q_.front());
        q_.pop();
        not_full_.notify_one();
        return val;
    }
};
```

---

## 总结

| 概念 | 要点 |
|------|------|
| `wait(lk, pred)` | 等待谓词为 true，自动处理虚假唤醒 |
| `notify_one` | 唤醒一个线程 |
| `notify_all` | 唤醒所有线程 |
| `unique_lock` 必须 | wait 需要解锁 → 重新加锁的能力 |
| 虚假唤醒 | 硬件原因可能自行醒来，务必用谓词版 wait |
