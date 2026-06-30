# 线程池

> 复用线程，高效管理并发任务

## 核心问题

**频繁创建/销毁线程 = 性能浪费**

```cpp
// ❌ 每次都创建新线程（慢）
for (int i = 0; i < 10000; ++i) {
    std::thread t([i]{ process(i); });
    t.join();
    // 创建线程开销大
}

// ✅ 用线程池（快）
ThreadPool pool(4);  // 创建 4 个工作线程
for (int i = 0; i < 10000; ++i) {
    pool.submit([i]{ process(i); });
    // 复用线程，无需创建
}
```

## 线程池原理

**核心思想**：
1. 预先创建固定数量的线程（工作线程）
2. 任务放入队列
3. 工作线程从队列取任务执行
4. 任务完成后，线程不销毁，继续等待新任务

**结构**：
```
┌─────────────────────────────────┐
│       ThreadPool                │
├─────────────────────────────────┤
│ 任务队列：[task1, task2, ...]  │
│ 工作线程：[t1, t2, t3, t4]     │
│ 互斥锁：保护队列                │
│ 条件变量：通知有新任务          │
└─────────────────────────────────┘
```

## 简单线程池实现

```cpp
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
    std::vector<std::thread> workers_;              // 工作线程
    std::queue<std::function<void()>> tasks_;       // 任务队列
    std::mutex queue_mutex_;                        // 保护队列
    std::condition_variable condition_;             // 通知新任务
    bool stop_;                                     // 停止标志

public:
    // 构造：创建 n 个工作线程
    ThreadPool(size_t threads) : stop_(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);

                        // 等待新任务或停止信号
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        // 停止且队列空，退出
                        if (stop_ && tasks_.empty()) return;

                        // 取任务
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    // 执行任务（不持有锁）
                    task();
                }
            });
        }
    }

    // 析构：停止所有线程
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();

        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    // 提交任务
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("线程池已停止");
            }
            tasks_.emplace([task]() { (*task)(); });
        }

        condition_.notify_one();
        return res;
    }
};
```

## 使用示例

### 基本使用

```cpp
ThreadPool pool(4);  // 4 个工作线程

// 提交任务
for (int i = 0; i < 10; ++i) {
    pool.submit([i] {
        std::cout << "任务 " << i << " 执行中\n";
    });
}

// 线程池自动分配线程执行
```

### 获取返回值

```cpp
ThreadPool pool(4);

// 提交有返回值的任务
std::vector<std::future<int>> results;

for (int i = 0; i < 10; ++i) {
    results.push_back(pool.submit([i] {
        return i * i;
    }));
}

// 获取结果
for (auto& result : results) {
    std::cout << result.get() << " ";
}
```

### 并行计算

```cpp
ThreadPool pool(std::thread::hardware_concurrency());

// 并行计算斐波那契数列
std::vector<std::future<int>> futures;

for (int i = 0; i < 40; ++i) {
    futures.push_back(pool.submit([i] {
        return fibonacci(i);
    }));
}

// 收集结果
for (auto& fut : futures) {
    std::cout << fut.get() << " ";
}
```

## 线程池的优势

**1. 性能**：
- 避免频繁创建/销毁线程
- 线程复用，开销小

**2. 资源控制**：
- 限制并发线程数
- 防止线程数量失控

**3. 管理方便**：
- 统一管理所有异步任务
- 自动负载均衡

## 线程池大小选择

**CPU 密集型任务**：
```cpp
// 线程数 = CPU 核心数
size_t threads = std::thread::hardware_concurrency();
ThreadPool pool(threads);
```

**I/O 密集型任务**：
```cpp
// 线程数 = CPU 核心数 × 2（或更多）
size_t threads = std::thread::hardware_concurrency() * 2;
ThreadPool pool(threads);
```

**混合任务**：
```cpp
// 根据实际情况调整，通过测试确定最优值
ThreadPool pool(8);
```

## 线程池 vs 直接创建线程

### 直接创建线程

```cpp
// ❌ 每次创建新线程
for (int i = 0; i < 1000; ++i) {
    std::thread t([i]{ process(i); });
    t.detach();  // 或 join()
}
```

**缺点**：
- 创建/销毁开销大
- 线程数失控
- 难以管理

### 线程池

```cpp
// ✅ 复用线程
ThreadPool pool(4);
for (int i = 0; i < 1000; ++i) {
    pool.submit([i]{ process(i); });
}
```

**优点**：
- 性能更好
- 线程数可控
- 易于管理

## 实际应用场景

**1. Web 服务器**：
```cpp
ThreadPool pool(100);  // 100 个工作线程

while (true) {
    auto request = accept_connection();
    pool.submit([request] {
        handle_request(request);
    });
}
```

**2. 图像处理**：
```cpp
ThreadPool pool(8);

std::vector<std::future<Image>> results;
for (auto& img : images) {
    results.push_back(pool.submit([&img] {
        return process_image(img);
    }));
}
```

**3. 数据处理**：
```cpp
ThreadPool pool(4);

for (auto& data_chunk : data) {
    pool.submit([&data_chunk] {
        process_data(data_chunk);
    });
}
```

## 常见陷阱

### 陷阱 1：忘记获取 future 结果

```cpp
// ❌ 忘记保存 future
ThreadPool pool(4);
for (int i = 0; i < 10; ++i) {
    pool.submit([i]{ return i * i; });  // future 被丢弃
}

// ✅ 保存 future
std::vector<std::future<int>> results;
for (int i = 0; i < 10; ++i) {
    results.push_back(pool.submit([i]{ return i * i; }));
}
```

### 陷阱 2：死锁（任务互相等待）

```cpp
// ❌ 死锁
ThreadPool pool(2);

auto fut1 = pool.submit([&pool] {
    auto fut2 = pool.submit([] { return 42; });
    return fut2.get();  // 等待 fut2，但线程池已满
});

fut1.get();  // 等待 fut1，但 fut1 在等待 fut2
```

**解决**：增加线程池大小，或避免嵌套提交

### 陷阱 3：任务数量过多，内存溢出

```cpp
// ❌ 任务队列无限增长
ThreadPool pool(4);
for (int i = 0; i < 10000000; ++i) {
    pool.submit([i]{ slow_task(i); });
    // 任务提交速度 > 执行速度，队列爆满
}

// ✅ 限制队列大小或分批提交
```

### 陷阱 4：线程池析构时有任务在执行

```cpp
// ❌ 提前析构
{
    ThreadPool pool(4);
    for (int i = 0; i < 100; ++i) {
        pool.submit([i]{ slow_task(i); });
    }
}  // pool 析构，但任务可能还未完成

// ✅ 确保任务完成
ThreadPool pool(4);
std::vector<std::future<void>> futures;
for (int i = 0; i < 100; ++i) {
    futures.push_back(pool.submit([i]{ slow_task(i); }));
}
for (auto& fut : futures) {
    fut.get();  // 等待所有任务完成
}
```

## 核心要点

1. **线程池 = 复用线程，避免频繁创建/销毁**
2. **结构**：任务队列 + 工作线程 + 互斥锁 + 条件变量
3. **使用 `packaged_task` + `future` 获取返回值**
4. **线程池大小**：
   - CPU 密集型 → CPU 核心数
   - I/O 密集型 → CPU 核心数 × 2
5. **优势**：性能好、资源可控、易管理
6. **注意**：
   - 避免任务互相等待（死锁）
   - 限制任务队列大小
   - 确保任务完成后再析构
