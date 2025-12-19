# 条件变量

> 线程间的等待/通知机制

## 核心问题：线程如何等待条件？

**忙等（错误做法）**：
```cpp
// ❌ 忙等：浪费 CPU
std::mutex mtx;
bool ready = false;

// 消费者线程
while (true) {
    std::lock_guard<std::mutex> lock(mtx);
    if (ready) break;
    // 不断循环检查，CPU 100%
}
```

**解决方案：条件变量**
```cpp
// ✅ 用条件变量：高效等待
std::mutex mtx;
std::condition_variable cv;
bool ready = false;

// 消费者线程
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return ready; });  // 阻塞等待，不占 CPU
}
```

## 基本用法

### 等待（wait）

```cpp
std::mutex mtx;
std::condition_variable cv;
bool ready = false;

// 等待线程
void wait_thread() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return ready; });  // 条件为真才继续
    // ready 为 true，继续执行
}
```

**wait 的两个参数**：
- 第1个：`unique_lock`（不能用 `lock_guard`）
- 第2个：谓词函数（返回 bool）

**wait 的行为**：
1. 检查条件：如果为真，立即返回
2. 如果为假：解锁 mutex，线程休眠（不占 CPU）
3. 被唤醒后：重新加锁，再次检查条件
4. 条件为真才返回，否则继续休眠

### 通知（notify）

```cpp
// 生产者线程
void notify_thread() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;  // 修改条件
    }  // 先解锁

    cv.notify_one();  // 唤醒一个等待的线程
    // 或 cv.notify_all();  // 唤醒所有等待的线程
}
```

**notify_one vs notify_all**：
- `notify_one()`：唤醒一个线程（单消费者）
- `notify_all()`：唤醒所有线程（多消费者）

## 为什么必须用 unique_lock？

```cpp
// ❌ 不能用 lock_guard
std::lock_guard<std::mutex> lock(mtx);
cv.wait(lock);  // 编译错误

// ✅ 必须用 unique_lock
std::unique_lock<std::mutex> lock(mtx);
cv.wait(lock);  // 正确
```

**原因**：
- `wait` 需要临时解锁（让其他线程修改条件）
- `lock_guard` 不支持解锁
- `unique_lock` 可以手动 lock/unlock

## 经典模式：生产者-消费者

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>

std::queue<int> buffer;
std::mutex mtx;
std::condition_variable cv;
const int MAX_SIZE = 10;

// 生产者
void producer() {
    for (int i = 0; i < 100; ++i) {
        std::unique_lock<std::mutex> lock(mtx);

        // 等待缓冲区不满
        cv.wait(lock, []{ return buffer.size() < MAX_SIZE; });

        buffer.push(i);
        std::cout << "生产: " << i << "\n";

        cv.notify_all();  // 通知消费者
    }
}

// 消费者
void consumer() {
    for (int i = 0; i < 100; ++i) {
        std::unique_lock<std::mutex> lock(mtx);

        // 等待缓冲区不空
        cv.wait(lock, []{ return !buffer.empty(); });

        int value = buffer.front();
        buffer.pop();
        std::cout << "消费: " << value << "\n";

        cv.notify_all();  // 通知生产者
    }
}
```

## wait 的三种形式

### 1. 带谓词（推荐）

```cpp
cv.wait(lock, []{ return ready; });
```

**等价于**：
```cpp
while (!ready) {
    cv.wait(lock);  // 不带谓词的 wait
}
```

### 2. 不带谓词（需要手动循环）

```cpp
// ❌ 错误：可能虚假唤醒
cv.wait(lock);
// ready 不一定为 true

// ✅ 正确：必须循环检查
while (!ready) {
    cv.wait(lock);
}
```

### 3. 带超时

```cpp
// 等待最多 1 秒
if (cv.wait_for(lock, std::chrono::seconds(1), []{ return ready; })) {
    // 条件满足
} else {
    // 超时
}

// 等待到指定时间点
auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
cv.wait_until(lock, deadline, []{ return ready; });
```

## 虚假唤醒

**问题**：`wait` 可能在条件不满足时被唤醒

```cpp
// ❌ 危险：不检查条件
cv.wait(lock);
// 假设 ready 为 true，但可能是虚假唤醒

// ✅ 安全：总是检查条件
cv.wait(lock, []{ return ready; });
// 或
while (!ready) {
    cv.wait(lock);
}
```

**原因**：
- 操作系统可能无故唤醒线程
- 多个线程竞争时，条件可能被其他线程改变

**解决**：总是用谓词或循环检查

## 通知时机

```cpp
// ✅ 好：先解锁再通知
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}  // 解锁
cv.notify_one();

// ⚠️ 可以但不推荐：持有锁时通知
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
    cv.notify_one();  // 等待线程被唤醒，但立即被锁阻塞
}
```

**建议**：先解锁再通知（性能更好）

## 常见陷阱

### 陷阱 1：忘记检查条件

```cpp
// ❌ 虚假唤醒导致错误
cv.wait(lock);
int value = buffer.front();  // buffer 可能为空

// ✅ 总是检查条件
cv.wait(lock, []{ return !buffer.empty(); });
int value = buffer.front();  // 安全
```

### 陷阱 2：用 lock_guard

```cpp
// ❌ 编译错误
std::lock_guard<std::mutex> lock(mtx);
cv.wait(lock);

// ✅ 必须用 unique_lock
std::unique_lock<std::mutex> lock(mtx);
cv.wait(lock);
```

### 陷阱 3：修改条件后不通知

```cpp
// ❌ 等待线程永远阻塞
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}
// 忘记 cv.notify_one()

// ✅ 修改后立即通知
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}
cv.notify_one();
```

### 陷阱 4：死锁

```cpp
// ❌ 死锁：持有锁时等待
void bad() {
    std::unique_lock<std::mutex> lock(mtx);
    // ... 做一些操作 ...
    other_function();  // 可能也尝试获取 mtx
}

// ✅ 需要时临时解锁
void good() {
    std::unique_lock<std::mutex> lock(mtx);
    // ... 做一些操作 ...
    lock.unlock();
    other_function();  // 不持有锁
    lock.lock();
}
```

## 核心要点

1. **条件变量用于线程间等待/通知**
2. **必须配合 `unique_lock` 使用**（不能用 `lock_guard`）
3. **总是用谓词检查条件**（避免虚假唤醒）
4. **修改条件后立即通知**
5. **先解锁再通知**（性能更好）
6. **`notify_one` vs `notify_all`**：
   - 单消费者 → `notify_one`
   - 多消费者 → `notify_all`
