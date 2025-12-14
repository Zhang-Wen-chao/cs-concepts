# 01 · Thread Basics

> 阶段 2 的第一个主题：用现代 C++ 正确地创建、管理和回收线程。

## 🎯 学习目标

1. 知道什么是线程、线程与进程的区别以及线程生命周期。
2. 能够使用 `std::thread` 创建线程，并理解 `join()` / `detach()` 的差别。
3. 能够在线程函数里接收参数（值传递、引用、移动语义、lambda）。
4. 了解 `std::thread::hardware_concurrency()`、`std::this_thread::sleep_for` 等辅助 API。
5. 意识到线程资源也需要 RAII（离开作用域前必须 `join` 或 `detach`）。

## 1. 线程是什么？

- **进程** 是操作系统分配资源的最小单位，拥有独立的虚拟地址空间。
- **线程** 是进程内的执行路径，同一个进程的线程共享地址空间、打开的文件等资源。
- 现代 CPU 通常有多个硬件线程（core × SMT），让我们有机会并行执行任务。
- 线程生命周期：创建 → 可运行/运行 → 阻塞/就绪 → 终止。

## 2. 第一个 `std::thread`

```cpp
#include <iostream>
#include <thread>

void SayHello() {
    std::cout << "Hello from worker thread!" << std::endl;
}

int main() {
    std::thread worker(SayHello);  // 创建并启动线程
    worker.join();                 // 等待线程结束
    std::cout << "Back in main" << std::endl;
}
```

关键点：
- 构造 `std::thread` 时，线程立即启动执行传入的可调用对象。
- 离开作用域前必须 `join()`（阻塞等待结束）或 `detach()`（交给运行时后台执行，自己不再关心）。
- 如果线程对象在未 `join`/`detach` 的情况下析构，程序会 `std::terminate()` —— 这是标准强制的安全措施。

## 3. `join()` vs `detach()`

| 方法  | 行为 | 适用场景 | 风险 |
|-------|------|---------|------|
| `join()`  | 当前线程阻塞，等待 worker 结束；回收其资源 | 绝大多数情况 | 如果忘记 `join`，程序异常终止 |
| `detach()`| 让 worker 变成“后台线程”，线程对象立即失效 | 后台任务、日志刷新等“不需要结果”场景 | 无法再与线程通信，线程里不得访问已经销毁的对象 |

**最佳实践：** 优先 `join()`；确实需要分离线程时，确保线程函数只访问长生命周期对象（例如单例、堆上共享数据）。

## 4. 线程函数如何接收参数？

```cpp
void Work(int id, const std::string& name) {
    std::cout << "Worker " << id << " => " << name << std::endl;
}

int main() {
    std::string task = "download";
    std::thread t(Work, 1, task);    // 参数按值复制
    std::thread t2(Work, 2, std::ref(task));  // std::ref 传引用

    t.join();
    t2.join();
}
```

- `std::thread` 的构造函数按值复制（或移动）参数。
- 希望在线程里修改外部变量，需要 `std::ref` / `std::cref`。
- 可以用 lambda 捕获外部变量，更直观：

```cpp
int counter = 0;
std::thread t([&counter] { counter += 1; });
```

## 5. `std::move` 与临时对象

- 线程函数如果需要接手一个昂贵对象，建议 `std::move` 进去，避免复制：

```cpp
std::vector<int> data(1'000'000, 42);
std::thread t(ProcessLargeData, std::move(data));
```

- `std::thread` 本身也可以被移动：`std::thread t2 = std::move(t);`，常用于存入容器。

## 6. 线程相关的其他 API

- `std::thread::hardware_concurrency()`：返回系统建议的并行线程数（可能为 0，表示未知）。
- `std::this_thread::get_id()`：获取当前线程 ID。
- `std::this_thread::sleep_for(duration)` / `sleep_until(time_point)`：让当前线程休眠。

```cpp
std::cout << "suggested threads: " << std::thread::hardware_concurrency() << std::endl;
```

## 7. RAII 管理线程

```cpp
class ThreadGuard {
public:
    explicit ThreadGuard(std::thread t) : t_(std::move(t)) {}
    ~ThreadGuard() {
        if (t_.joinable()) {
            t_.join();
        }
    }

private:
    std::thread t_;
};
```

- `joinable()` 用来判断线程对象是否仍然拥有底层线程。
- RAII 守卫能避免异常或早退导致忘记 `join`。

## 8. 常见坑

1. **引用悬空**：线程里访问了已销毁的局部变量。
2. **重复 `join()`**：同一个线程只能 `join` 一次，第二次会抛异常。
3. **忙等**：线程里自旋等待事件而不 `sleep`，导致 CPU 飙升。
4. **线程数量失控**：无节制地创建线程，频繁上下文切换反而更慢 → 后续会用线程池解决。

## 9. 实践：多线程累加器

对应代码：`practices/01_thread_basics.cpp`

任务描述：
1. 有一个 `std::vector<int>`，里面是 1..1_000_000。
2. 启动 `std::thread::hardware_concurrency()` 个线程，每个线程处理一段区间，计算部分和。
3. 主线程收集部分和，得到总和，并验证结果是否等于公式 `n*(n+1)/2`。
4. 打印每个线程的 ID 和它处理的区间，并测量总体用时。

**思考题：**
- 如果忘了 `join()` 会发生什么？
- 如果把线程数设置得远大于 CPU 数，会怎样？
- 如何把 ThreadGuard 改成可以 `join` 或 `detach` 的可配置 RAII？

---

继续学习顺序：
1. 通关本章代码。
2. 阅读 `02_mutex_locks.md`，学习互斥锁/锁守卫。
3. 实践生产者-消费者模型，为后面做线程池铺路。
