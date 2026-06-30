# std::mutex、lock_guard、unique_lock、scoped_lock

## 1. 数据竞争与互斥锁

多个线程写同一数据 → 数据竞争（undefined behavior）→ 需要互斥锁：

```cpp
#include <iostream>
#include <thread>
#include <mutex>

int counter = 0;
std::mutex mtx;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        mtx.lock();
        ++counter;          // 临界区
        mtx.unlock();
    }
}

int main() {
    std::thread t1(increment), t2(increment);
    t1.join(); t2.join();
    std::cout << counter << '\n';  // 200000
}
```

**手动 lock/unlock 的坑**：如果临界区抛出异常，`unlock` 不会执行 → 死锁。

---

## 2. std::lock_guard — RAII 锁

构造时加锁，析构时自动解锁：

```cpp
#include <mutex>

std::mutex mtx;

void safe_increment(int& counter) {
    std::lock_guard<std::mutex> lock(mtx);  // 加锁
    ++counter;                               // 临界区安全
}  // 析构时自动 unlock
```

**最佳实践**：所有加锁优先用 `lock_guard`，除非需要手动解锁或条件锁。

---

## 3. std::unique_lock — 更灵活的 RAII 锁

`unique_lock` 相比 `lock_guard` 多了：

| 功能 | 说明 |
|------|------|
| `lock()` / `unlock()` | 可手动控制 |
| 延迟加锁 | `unique_lock<mutex> lk(mtx, std::defer_lock)` |
| 转移所有权 | 移动语义 |
| 配合条件变量 | `wait()` 需要 `unique_lock` |

```cpp
#include <mutex>

std::mutex mtx;

void example() {
    std::unique_lock<std::mutex> lk(mtx);           // 立即加锁
    // ... 做点事 ...
    lk.unlock();                                     // 提前解锁
    // ... 不需要锁的耗时操作 ...
    lk.lock();                                       // 重新加锁
}  // 析构时如果还持有锁则自动解锁

void defer_example() {
    std::unique_lock<std::mutex> lk(mtx, std::defer_lock);  // 不立即加锁
    lk.lock();   // 手动加锁
}
```

---

## 4. std::scoped_lock — 死锁预防（C++17）

一次性锁多个 mutex，使用**死锁避免算法**（类似 std::lock）：

```cpp
#include <mutex>

std::mutex m1, m2;

void transfer(int& from, int& to, int amount) {
    // scoped_lock 一次锁两个，不会有死锁
    std::scoped_lock lock(m1, m2);
    from -= amount;
    to += amount;
}
```

**注意**：`scoped_lock` 等价于 `lock_guard` 的多锁版本，C++17 起推荐用它代替 `lock_guard`。

---

## 5. 死锁的四个条件

1. **互斥**：资源一次只能被一个线程持有
2. **持有并等待**：线程持有一个资源，等待另一个
3. **不可剥夺**：资源只能由持有者主动释放
4. **循环等待**：A 等 B 占有的资源，B 等 A 占有的资源

**预防**：永远用一致的加锁顺序，或直接用 `std::scoped_lock`。

---

## 总结

| 锁类型 | 特性 | 引入 |
|--------|------|------|
| `lock_guard` | 简单 RAII，不可手动 unlock | C++11 |
| `unique_lock` | 灵活 RAII，可手动解锁 | C++11 |
| `scoped_lock` | 多锁死锁预防 | C++17 |
