# 线程基础

> C++11 引入的标准线程库

## 核心概念

**线程 = 独立的执行流，共享进程的内存空间**

```cpp
#include <thread>

int main() {
    std::thread t([]{ std::cout << "子线程\n"; });  // 创建线程
    std::cout << "主线程\n";
    t.join();  // 等待子线程结束
}
// 输出顺序不确定（并发执行）
```

## 创建线程的三种方式

```cpp
// 1. 函数
void work() { std::cout << "工作中\n"; }
std::thread t1(work);

// 2. Lambda（推荐）
std::thread t2([]{
    std::cout << "工作中\n";
});

// 3. 函数对象
struct Worker {
    void operator()() { std::cout << "工作中\n"; }
};
std::thread t3(Worker{});
```

## join vs detach

### join：等待线程结束

```cpp
std::thread t([]{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "子线程完成\n";
});

std::cout << "等待中...\n";
t.join();  // 阻塞，直到 t 完成
std::cout << "继续执行\n";
```

### detach：分离线程

```cpp
std::thread t([]{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "子线程\n";  // 可能不会执行
});

t.detach();  // 分离，不再等待
std::cout << "主线程结束\n";
// 主线程可能在子线程完成前退出
```

**⚠️ detach 的危险**：主线程退出 → 进程终止 → 子线程被强制结束

## 传递参数

```cpp
void task(int x, const std::string& s) {
    std::cout << x << " " << s << "\n";
}

// 按值传递
std::thread t1(task, 42, "hello");

// 引用传递（必须用 std::ref）
int n = 10;
std::thread t2([](int& x) { x = 20; }, std::ref(n));
t2.join();
std::cout << n;  // 20
```

**为什么要 `std::ref`？**
- `std::thread` 内部会拷贝所有参数
- `std::ref` 告诉它"传引用"

## 移动语义

```cpp
// 移动大对象（避免拷贝）
std::vector<int> data(1000000);
std::thread t(process, std::move(data));

// 线程本身也可移动
std::vector<std::thread> threads;
threads.push_back(std::move(t));
```

## 线程信息

```cpp
// 硬件并发数（CPU 核心数）
std::cout << std::thread::hardware_concurrency() << "\n";

// 线程 ID
std::thread t([]{ std::cout << std::this_thread::get_id() << "\n"; });

// 休眠
std::this_thread::sleep_for(std::chrono::seconds(1));
std::this_thread::sleep_for(std::chrono::milliseconds(100));
```

## 常见陷阱

### 陷阱 1：忘记 join 或 detach

```cpp
// ❌ 危险
void bad() {
    std::thread t([]{ /* ... */ });
    // 离开作用域，t 析构
    // 既没 join 也没 detach → 调用 std::terminate，程序崩溃
}

// ✅ 正确
void good() {
    std::thread t([]{ /* ... */ });
    t.join();  // 或 t.detach()
}
```

### 陷阱 2：引用捕获局部变量

```cpp
// ❌ 危险
void bad() {
    int x = 10;
    std::thread t([&x]{
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << x;  // x 已销毁，悬空引用
    });
    t.detach();
    // 函数结束，x 销毁，但线程还在运行
}

// ✅ 正确：按值捕获
void good() {
    int x = 10;
    std::thread t([x]{
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << x;  // 安全，拷贝了 x
    });
    t.detach();
}
```

### 陷阱 3：重复 join

```cpp
// ❌ 危险
std::thread t([]{ /* ... */ });
t.join();
t.join();  // 崩溃：不能重复 join
```

## RAII 线程管理

**问题**：手动 join 容易忘记，异常时也会跳过

```cpp
// ✅ RAII 包装
class ThreadGuard {
    std::thread& t_;
public:
    explicit ThreadGuard(std::thread& t) : t_(t) {}
    ~ThreadGuard() {
        if (t_.joinable()) t_.join();
    }

    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// 使用
std::thread t([]{ /* ... */ });
ThreadGuard guard(t);  // 析构时自动 join
```

**更简单**（C++20）：
```cpp
std::jthread t([]{ /* ... */ });  // 析构时自动 join
```

## 核心要点

1. **线程创建后必须 join 或 detach**，否则程序崩溃
2. **join：等待结束**（常用），**detach：分离运行**（慎用）
3. **按值捕获局部变量**（detach 时）
4. **引用传递用 `std::ref`**
5. **用 RAII 管理线程**（C++20 用 `std::jthread`）
6. **线程数 ≈ CPU 核心数**（过多会性能下降）
