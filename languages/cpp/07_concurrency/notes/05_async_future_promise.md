# std::async、future/promise、packaged_task
> For a detailed walkthrough with more examples, see `_supplementary/05_async_future.md`

## 1. std::async — 异步任务的最简单方式

把函数丢到后台跑，用 `std::future` 拿结果：

```cpp
#include <iostream>
#include <future>

int heavy_compute(int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += i;
    return sum;
}

int main() {
    // std::launch::async → 立即在新线程执行
    auto fut = std::async(std::launch::async, heavy_compute, 10000);

    // 做点别的事...
    std::cout << "waiting...\n";

    // 获取结果（阻塞直到完成）
    int result = fut.get();
    std::cout << "result = " << result << '\n';
}
```

**启动策略**：

| 策略 | 含义 |
|------|------|
| `std::launch::async` | 强制在新线程执行 |
| `std::launch::deferred` | 延迟执行，`get()` 时才在调用线程执行 |
| 默认 | 由实现决定（可能是 async，也可能是 deferred） |

---

## 2. std::future — 未来值

`future<T>` 持有**一个**值，只能 `get()` 一次（`get()` 后 future 无效）：

```cpp
#include <future>
#include <iostream>

int main() {
    std::promise<int> prom;                  // 生产端
    std::future<int> fut = prom.get_future();// 消费端

    std::thread t([&prom] {
        prom.set_value(42);                   // 生产值
    });

    int val = fut.get();                      // 消费值
    std::cout << val << '\n';
    t.join();
}
```

**几种 future**：

| 类型 | 说明 |
|------|------|
| `std::future<T>` | 唯一消费者 |
| `std::shared_future<T>` | 可复制，多个线程可同时 `get()` |
| `std::future<void>` | 无返回值的异步任务 |

---

## 3. std::promise — 手动设值

`promise` 和 `future` 是一对。`promise` 负责设值，`future` 负责取值：

```cpp
#include <iostream>
#include <future>
#include <thread>
#include <vector>

void worker(std::promise<int> prom) {
    try {
        // ... 可能抛异常 ...
        prom.set_value(42);
    } catch (...) {
        prom.set_exception(std::current_exception());  // 异常传给 future
    }
}

int main() {
    std::promise<int> prom;
    auto fut = prom.get_future();

    std::thread t(worker, std::move(prom));
    t.detach();

    try {
        std::cout << fut.get() << '\n';
    } catch (const std::exception& e) {
        std::cout << "caught: " << e.what() << '\n';
    }
}
```

**注意**：`promise` 不可复制，只能移动。

---

## 4. std::packaged_task — 包装可调用对象

把函数/仿函数/lambda 包装起来，它的 future 会持有返回值：

```cpp
#include <iostream>
#include <future>
#include <thread>

int add(int a, int b) { return a + b; }

int main() {
    std::packaged_task<int(int, int)> task(add);
    auto fut = task.get_future();

    // 在新线程执行打包的任务
    std::thread t(std::move(task), 1, 2);
    t.join();

    std::cout << fut.get() << '\n';  // 3
}
```

**`packaged_task` 可复用的地方**：可以多次 `()` 调用（但只有第一次调用的结果能通过 future 获取）。

---

## 5. 三者对比

| 组件 | 角色 | 适用场景 |
|------|------|---------|
| `std::async` | 高层 API | 简单异步任务，不关心线程管理 |
| `std::promise` / `future` | 值传递通道 | 手动控制何时设值 |
| `std::packaged_task` | 包装可调用对象 | 任务队列/线程池 |

**选择建议**：
- 能用 `async` 就不用 `promise`
- 需要手动管理线程生命周期 → `packaged_task`
- 需要在不同线程之间传递值 → `promise` / `future`

---

## 总结

| 特性 | 关键点 |
|------|--------|
| `async` | 最简单，结果通过 future 拿 |
| `future::get()` | 阻塞 + 只能调一次 |
| `shared_future` | 多个线程同时读 |
| `promise` | 手动设值/异常 |
| `packaged_task` | 包装可调用对象 |
