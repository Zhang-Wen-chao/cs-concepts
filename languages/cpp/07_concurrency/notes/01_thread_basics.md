# std::thread 基础、join/detach、jthread（C++20）
> For a detailed walkthrough with more examples, see `_supplementary/01_thread_basics.md`

## 1. std::thread — 开启新线程

```cpp
#include <iostream>
#include <thread>

void hello() {
    std::cout << "Hello from thread! tid = "
              << std::this_thread::get_id() << '\n';
}

int main() {
    std::thread t(hello);  // 启动新线程执行 hello()
    t.join();              // 等待线程结束

    // 带参数的线程
    auto work = [](int n, const std::string& msg) {
        std::cout << "param: " << n << ", " << msg << '\n';
    };
    std::thread t2(work, 42, "answer");
    t2.join();
}
```

**线程参数按值传**：如果需要引用，必须用 `std::ref()` 包裹，否则会被拷贝。

```cpp
void foo(int& x) { x += 1; }

int main() {
    int a = 0;
    std::thread t(foo, std::ref(a));  // 需要 std::ref
    t.join();
    std::cout << a << '\n';  // 1
}
```

---

## 2. join vs detach

| 操作 | 行为 | 后果 |
|------|------|------|
| `t.join()` | 阻塞当前线程，等待 `t` 结束 | `t` 结束后可安全销毁 |
| `t.detach()` | 分离线程，让它在后台继续运行 | 无法再 `join`，`t` 不再可 join |

```cpp
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::thread t([]{
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "background done\n";
    });

    t.detach();           // 后台运行，主线程不等待
    // t.join();          // ❌ crash：detach 后不能 join

    std::cout << "main continues\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
}
```

**核心规则**：每个 `std::thread` 析构前必须 `join()` 或 `detach()`，否则 `std::terminate()`。

---

## 3. jthread — C++20 自动 join

`std::jthread` 在析构时自动 `join()`，还支持请求停止：

```cpp
#include <iostream>
#include <thread>

int main() {
    std::jthread t([](std::stop_token st) {
        while (!st.stop_requested()) {
            std::cout << "working...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::cout << "stopped gracefully\n";
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));
    t.request_stop();          // 请求停止
    // 自动 join，无需手动调用
}
```

**不再需要**手动 try-catch 包裹线程体来确保 join。

---

## 4. 线程 ID 与硬件并发

```cpp
#include <iostream>
#include <thread>

int main() {
    std::cout << "hardware concurrency: "
              << std::thread::hardware_concurrency() << '\n';

    auto main_id = std::this_thread::get_id();
    std::cout << "main thread id: " << main_id << '\n';
}
```

---

## 总结

| 特性 | 引入 | 关键点 |
|------|------|--------|
| `std::thread` | C++11 | 必须 join/detach |
| `std::jthread` | C++20 | 自动 join + stop_token |
| `hardware_concurrency()` | C++11 | 逻辑 CPU 核心数 |
| `std::ref()` | C++11 | 线程传引用时使用 |
