# 异步编程

> 简化异步任务的高级接口

## 核心概念

**异步 = 不等待结果，继续做其他事**

```cpp
// ❌ 同步：等待结果
int result = compute();  // 阻塞，等待计算完成
process(result);

// ✅ 异步：不等待，继续执行
std::future<int> fut = std::async(compute);  // 立即返回
// ... 做其他事 ...
int result = fut.get();  // 需要时再获取结果
process(result);
```

## std::async - 启动异步任务

**基本用法**：
```cpp
#include <future>

// 启动异步任务
std::future<int> fut = std::async([]{
    return 42;
});

// 获取结果（阻塞，直到任务完成）
int result = fut.get();
```

**两种启动策略**：
```cpp
// 1. std::launch::async - 立即创建新线程
auto fut1 = std::async(std::launch::async, task);

// 2. std::launch::deferred - 延迟执行（调用 get 时才执行）
auto fut2 = std::async(std::launch::deferred, task);

// 3. 默认 - 由实现决定（可能是 async 或 deferred）
auto fut3 = std::async(task);
```

## std::future - 获取结果

```cpp
std::future<int> fut = std::async([]{ return 42; });

// 获取结果（只能调用一次）
int result = fut.get();  // 阻塞，直到任务完成
// fut.get();  // ❌ 错误：不能重复调用

// 检查状态
if (fut.valid()) {
    // future 有效，可以调用 get
}

// 等待（不获取结果）
fut.wait();  // 阻塞，直到任务完成

// 等待一段时间
auto status = fut.wait_for(std::chrono::seconds(1));
if (status == std::future_status::ready) {
    // 任务完成
} else if (status == std::future_status::timeout) {
    // 超时
}
```

## std::promise - 手动设置结果

**promise 和 future 是一对**：
- `promise`：生产者，设置结果
- `future`：消费者，获取结果

```cpp
std::promise<int> prom;
std::future<int> fut = prom.get_future();

// 生产者线程
std::thread t([&prom]{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    prom.set_value(42);  // 设置结果
});

// 消费者线程
int result = fut.get();  // 阻塞，直到 promise 设置值
std::cout << result;  // 42

t.join();
```

**设置异常**：
```cpp
std::promise<int> prom;
std::future<int> fut = prom.get_future();

std::thread t([&prom]{
    try {
        throw std::runtime_error("错误");
    } catch (...) {
        prom.set_exception(std::current_exception());
    }
});

try {
    fut.get();  // 抛出异常
} catch (const std::exception& e) {
    std::cout << e.what();  // "错误"
}

t.join();
```

## std::packaged_task - 包装函数

**packaged_task = 可调用对象 + promise/future**

```cpp
// 包装函数
std::packaged_task<int(int, int)> task([](int a, int b) {
    return a + b;
});

// 获取 future
std::future<int> fut = task.get_future();

// 在线程中执行
std::thread t(std::move(task), 10, 20);

// 获取结果
int result = fut.get();  // 30

t.join();
```

**用途：线程池**
```cpp
// 提交任务到线程池
template<typename F, typename... Args>
auto submit(F&& f, Args&&... args) {
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<ReturnType> result = task->get_future();

    // 添加到任务队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        tasks_.emplace([task]{ (*task)(); });
    }

    return result;
}
```

## std::shared_future - 多个消费者

**问题：future 只能 get 一次**
```cpp
std::future<int> fut = std::async([]{ return 42; });
int r1 = fut.get();
// int r2 = fut.get();  // ❌ 错误
```

**解决：shared_future 可以多次 get**
```cpp
std::future<int> fut = std::async([]{ return 42; });
std::shared_future<int> sf = fut.share();  // 转换

// 多个线程都可以获取结果
std::thread t1([sf]{ std::cout << sf.get(); });
std::thread t2([sf]{ std::cout << sf.get(); });
std::thread t3([sf]{ std::cout << sf.get(); });

t1.join();
t2.join();
t3.join();
```

## 实际应用：并行计算

```cpp
// 并行计算斐波那契数列
std::vector<std::future<int>> futures;

for (int i = 0; i < 10; ++i) {
    futures.push_back(std::async(std::launch::async, [i]{
        // 计算第 i 个斐波那契数
        return fibonacci(i);
    }));
}

// 收集结果
std::vector<int> results;
for (auto& fut : futures) {
    results.push_back(fut.get());
}
```

## async vs thread

**async（推荐）**：
```cpp
// ✅ 简洁，自动管理线程
auto fut = std::async([]{ return 42; });
int result = fut.get();
```

**优点**：
- 代码简洁
- 自动管理线程生命周期
- 返回值通过 future 传递
- 异常自动传播

**thread（底层）**：
```cpp
// ❌ 复杂，手动管理
int result;
std::thread t([&result]{ result = 42; });
t.join();
```

**缺点**：
- 需要手动 join
- 返回值需要外部变量
- 异常处理复杂

**选择**：
- 简单异步任务 → `async`
- 需要精确控制线程 → `thread`
- 长期运行的后台线程 → `thread`

## 常见陷阱

### 陷阱 1：future 析构会阻塞

```cpp
// ❌ 阻塞主线程
{
    auto fut = std::async(std::launch::async, long_task);
    // ... 做其他事 ...
}  // fut 析构，阻塞等待任务完成

// ✅ 明确获取结果
auto fut = std::async(std::launch::async, long_task);
// ... 做其他事 ...
int result = fut.get();  // 明确等待
```

### 陷阱 2：忘记 get，任务不执行

```cpp
// ❌ deferred 策略，任务不执行
auto fut = std::async(std::launch::deferred, task);
// ... 做其他事 ...
// 忘记调用 fut.get()，task 永远不执行

// ✅ 明确调用 get
auto fut = std::async(std::launch::deferred, task);
fut.get();  // 执行任务
```

### 陷阱 3：重复 get

```cpp
// ❌ 不能重复 get
std::future<int> fut = std::async([]{ return 42; });
int r1 = fut.get();
// int r2 = fut.get();  // 崩溃

// ✅ 用 shared_future
std::shared_future<int> sf = std::async([]{ return 42; }).share();
int r1 = sf.get();
int r2 = sf.get();  // 正确
```

### 陷阱 4：promise 忘记设置值

```cpp
// ❌ promise 销毁前未设置值
{
    std::promise<int> prom;
    std::future<int> fut = prom.get_future();
    // prom 析构，fut.get() 会抛异常
}

// ✅ 确保设置值
std::promise<int> prom;
std::future<int> fut = prom.get_future();
prom.set_value(42);
fut.get();  // 正确
```

## 核心要点

1. **`async` - 启动异步任务**（最简单）
2. **`future` - 获取结果**（get 只能调用一次）
3. **`promise` - 手动设置结果**（配合 future）
4. **`packaged_task` - 包装函数**（用于线程池）
5. **`shared_future` - 多个消费者**（可以多次 get）
6. **选择**：
   - 简单异步 → `async`
   - 精确控制 → `thread`
   - 手动控制结果 → `promise`
   - 线程池 → `packaged_task`
7. **注意**：future 析构会阻塞，记得调用 get
