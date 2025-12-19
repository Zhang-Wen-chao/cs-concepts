# 互斥锁与 RAII 锁管理

> 多线程访问共享数据的安全机制

## 核心问题：数据竞争

**多个线程同时修改同一数据 → 结果不可预测**

```cpp
// ❌ 危险：数据竞争
int counter = 0;

std::thread t1([&]{ for(int i=0; i<100000; ++i) counter++; });
std::thread t2([&]{ for(int i=0; i<100000; ++i) counter++; });

t1.join();
t2.join();

std::cout << counter;  // 应该是 200000，实际可能是 150000（丢失更新）
```

**为什么？**
- `counter++` 不是原子操作
- 实际分三步：读取 → 加1 → 写回
- 两个线程可能同时读到相同的值

## 解决方案：互斥锁（mutex）

```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

std::thread t1([&]{
    for(int i=0; i<100000; ++i) {
        mtx.lock();      // 加锁
        counter++;
        mtx.unlock();    // 解锁
    }
});

std::thread t2([&]{
    for(int i=0; i<100000; ++i) {
        mtx.lock();
        counter++;
        mtx.unlock();
    }
});

t1.join();
t2.join();

std::cout << counter;  // 一定是 200000
```

**mutex 保证**：同一时刻只有一个线程能持有锁

## 手动 lock/unlock 的问题

```cpp
// ❌ 危险：忘记 unlock
mtx.lock();
if (error) return;  // 忘记 unlock，其他线程永远等待（死锁）
counter++;
mtx.unlock();

// ❌ 危险：异常时未 unlock
mtx.lock();
process();  // 如果抛异常，跳过 unlock
mtx.unlock();
```

## 解决方案：RAII 锁管理

### lock_guard（推荐，最常用）

```cpp
std::mutex mtx;
int counter = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mtx);  // 构造时加锁
    counter++;
    // 离开作用域自动解锁，即使有异常
}
```

**特点**：
- 构造时加锁，析构时解锁
- 不能手动 unlock
- 最轻量，性能最好

### unique_lock（更灵活）

```cpp
std::mutex mtx;

void foo() {
    std::unique_lock<std::mutex> lock(mtx);  // 加锁

    // 可以手动解锁
    lock.unlock();

    // ... 做其他不需要锁的事 ...

    // 可以再次加锁
    lock.lock();

    // 离开作用域自动解锁
}
```

**特点**：
- 可以手动 lock/unlock
- 可以转移所有权（移动）
- 支持条件变量（下一章）

### scoped_lock（C++17，多个锁）

```cpp
std::mutex mtx1, mtx2;

void transfer(Account& from, Account& to, int amount) {
    // 同时锁定两个互斥锁，避免死锁
    std::scoped_lock lock(mtx1, mtx2);

    from.balance -= amount;
    to.balance += amount;
}
```

**特点**：
- 可以同时锁定多个 mutex
- 自动避免死锁
- C++17 新增

## 三种锁的选择

```cpp
lock_guard     // 90% 情况，简单场景
unique_lock    // 需要手动控制，或配合条件变量
scoped_lock    // 需要同时锁定多个 mutex（C++17）
```

## 死锁问题

### 死锁示例

```cpp
// ❌ 死锁
std::mutex mtx1, mtx2;

// 线程 1
void thread1() {
    std::lock_guard<std::mutex> lock1(mtx1);  // 持有 mtx1
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard<std::mutex> lock2(mtx2);  // 等待 mtx2（被线程2持有）
}

// 线程 2
void thread2() {
    std::lock_guard<std::mutex> lock2(mtx2);  // 持有 mtx2
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard<std::mutex> lock1(mtx1);  // 等待 mtx1（被线程1持有）
}
// 互相等待，永远阻塞
```

### 避免死锁的方法

**方法 1：固定加锁顺序**
```cpp
// ✅ 总是按相同顺序加锁
void thread1() {
    std::lock_guard<std::mutex> lock1(mtx1);  // 先 mtx1
    std::lock_guard<std::mutex> lock2(mtx2);  // 后 mtx2
}

void thread2() {
    std::lock_guard<std::mutex> lock1(mtx1);  // 先 mtx1
    std::lock_guard<std::mutex> lock2(mtx2);  // 后 mtx2
}
```

**方法 2：用 scoped_lock（C++17）**
```cpp
// ✅ 自动避免死锁
void safe_transfer() {
    std::scoped_lock lock(mtx1, mtx2);  // 原子地锁定两个
    // ...
}
```

**方法 3：用 std::lock + unique_lock**
```cpp
// ✅ C++11 方式
std::unique_lock<std::mutex> lock1(mtx1, std::defer_lock);  // 不立即加锁
std::unique_lock<std::mutex> lock2(mtx2, std::defer_lock);
std::lock(lock1, lock2);  // 原子地同时加锁
```

## 性能建议

```cpp
// ✅ 好：锁的范围尽量小
{
    std::lock_guard<std::mutex> lock(mtx);
    data.push_back(value);  // 只锁保护关键操作
}
expensive_computation();  // 不需要锁

// ❌ 坏：锁的范围太大
{
    std::lock_guard<std::mutex> lock(mtx);
    data.push_back(value);
    expensive_computation();  // 浪费时间，其他线程等待
}
```

## 常见陷阱

### 陷阱 1：忘记加锁

```cpp
// ❌ 忘记加锁
std::mutex mtx;
int counter = 0;

void increment() {
    counter++;  // 数据竞争
}
```

### 陷阱 2：锁的粒度太大

```cpp
// ❌ 整个函数都锁住
void process() {
    std::lock_guard<std::mutex> lock(mtx);
    read_data();      // 需要锁
    compute();        // 不需要锁，但被锁住了
    write_result();   // 需要锁
}
```

### 陷阱 3：返回被保护数据的引用

```cpp
// ❌ 危险：锁解除后，引用仍然可以访问
class Data {
    std::mutex mtx;
    std::vector<int> vec;
public:
    std::vector<int>& get_data() {
        std::lock_guard<std::mutex> lock(mtx);
        return vec;  // 锁解除，但引用还在外面被用
    }
};
```

## 核心要点

1. **多线程访问共享数据必须加锁**
2. **优先用 `lock_guard`**（90%情况）
3. **锁的范围尽量小**（性能）
4. **固定加锁顺序**（避免死锁）
5. **用 RAII 管理锁**（永远不要手动 lock/unlock）
6. **多个锁用 `scoped_lock`**（C++17）
