# RAII 原则详解

> Resource Acquisition Is Initialization（资源获取即初始化）
> 更好的理解：**资源的生命周期绑定到对象的生命周期**

## 🎯 本课目标

- 深入理解 RAII 的工作原理
- 学会自己写 RAII 类
- 掌握 RAII 的最佳实践
- 理解标准库中的 RAII 实现

---

## 1️⃣ RAII 的工作原理

### C++ 的保证

**C++ 语言级别的保证：**
> 当对象离开作用域时，编译器**一定**会调用析构函数

```cpp
void foo() {
    {
        MyClass obj;  // 构造函数被调用
        // ...
    }  // ← 离开作用域，析构函数一定被调用
}
```

**即使有异常也会调用：**
```cpp
void foo() {
    MyClass obj;  // 构造函数

    throw std::runtime_error("error");  // 抛异常

}  // ← 即使有异常，obj 的析构函数也会被调用
```

**这就是 RAII 的基础！**

---

## 2️⃣ 自己实现一个 RAII 类

### 例子 1：文件管理类

```cpp
#include <cstdio>
#include <stdexcept>
#include <string>

class FileHandle {
public:
    // 构造函数：获取资源
    explicit FileHandle(const std::string& filename, const char* mode = "r") {
        file_ = std::fopen(filename.c_str(), mode);
        if (!file_) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }

    // 析构函数：释放资源
    ~FileHandle() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    // 禁止拷贝（防止重复释放）
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // 允许移动（转移所有权）
    FileHandle(FileHandle&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;  // 转移后，other 不再拥有资源
    }

    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            // 先释放自己的资源
            if (file_) {
                std::fclose(file_);
            }
            // 转移所有权
            file_ = other.file_;
            other.file_ = nullptr;
        }
        return *this;
    }

    // 提供访问文件的接口
    FILE* get() const { return file_; }

    // 读取一行
    bool read_line(std::string& line) {
        char buffer[1024];
        if (std::fgets(buffer, sizeof(buffer), file_)) {
            line = buffer;
            return true;
        }
        return false;
    }

private:
    FILE* file_ = nullptr;
};

// 使用示例
void read_config() {
    FileHandle file("config.txt");  // 自动打开

    std::string line;
    while (file.read_line(line)) {
        // 处理每一行
    }

}  // 自动关闭文件，即使有异常
```

**关键点：**
1. ✅ 构造时获取资源
2. ✅ 析构时释放资源
3. ✅ 禁止拷贝（`= delete`）
4. ✅ 允许移动（转移所有权）
5. ✅ 异常安全（析构函数不抛异常）

---

### 例子 2：锁管理类（模拟 std::lock_guard）

```cpp
#include <mutex>

template<typename Mutex>
class LockGuard {
public:
    // 构造时加锁
    explicit LockGuard(Mutex& mutex) : mutex_(mutex) {
        mutex_.lock();
    }

    // 析构时解锁
    ~LockGuard() {
        mutex_.unlock();
    }

    // 禁止拷贝和移动
    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;
    LockGuard(LockGuard&&) = delete;
    LockGuard& operator=(LockGuard&&) = delete;

private:
    Mutex& mutex_;
};

// 使用示例
std::mutex mtx;
int shared_data = 0;

void increment() {
    LockGuard<std::mutex> lock(mtx);  // 自动加锁

    ++shared_data;  // 操作共享数据

    if (shared_data > 100) {
        return;  // 提前返回，自动解锁
    }

}  // 离开作用域，自动解锁
```

---

### 例子 3：内存管理类（模拟 std::unique_ptr）

```cpp
template<typename T>
class UniquePtr {
public:
    // 构造：获取指针
    explicit UniquePtr(T* ptr = nullptr) : ptr_(ptr) {}

    // 析构：释放内存
    ~UniquePtr() {
        delete ptr_;
    }

    // 禁止拷贝
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    // 允许移动
    UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr_;  // 释放自己的资源
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // 解引用
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }

    // 获取原始指针
    T* get() const { return ptr_; }

    // 释放所有权
    T* release() {
        T* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }

private:
    T* ptr_;
};

// 使用示例
void foo() {
    UniquePtr<int> p(new int(42));  // 自动管理

    std::cout << *p << std::endl;  // 解引用

}  // 自动 delete，不会泄漏
```

---

## 3️⃣ RAII 的五大规则（Rule of Five）

当你的类管理资源时，需要定义这 5 个函数：

```cpp
class MyResource {
public:
    // 1. 析构函数
    ~MyResource() {
        // 释放资源
    }

    // 2. 拷贝构造函数
    MyResource(const MyResource& other) {
        // 深拷贝资源
    }

    // 3. 拷贝赋值运算符
    MyResource& operator=(const MyResource& other) {
        if (this != &other) {
            // 释放自己的资源
            // 深拷贝 other 的资源
        }
        return *this;
    }

    // 4. 移动构造函数
    MyResource(MyResource&& other) noexcept {
        // 转移资源所有权
        // 将 other 置为空状态
    }

    // 5. 移动赋值运算符
    MyResource& operator=(MyResource&& other) noexcept {
        if (this != &other) {
            // 释放自己的资源
            // 转移 other 的资源
            // 将 other 置为空状态
        }
        return *this;
    }
};
```

**简化版：Rule of Zero**
> 如果可以用标准库（vector、unique_ptr 等），就不要自己管理资源

```cpp
// ❌ Rule of Five：复杂
class MyClass {
    int* data_;
    // 需要定义 5 个函数...
};

// ✅ Rule of Zero：简单
class MyClass {
    std::vector<int> data_;  // 标准库自动管理
    // 不需要定义任何特殊函数！
};
```

---

## 4️⃣ RAII 的最佳实践

### 实践 1：永远用 RAII 管理资源

```cpp
// ❌ 不要这样
void bad_example() {
    int* data = new int[1000];

    // ... 处理数据 ...

    delete[] data;  // 容易忘记
}

// ✅ 这样做
void good_example() {
    std::vector<int> data(1000);

    // ... 处理数据 ...

}  // 自动释放
```

### 实践 2：资源获取就是初始化

```cpp
// ❌ 不要分两步
class Bad {
public:
    Bad() {}  // 构造函数不获取资源
    void init() { /* 获取资源 */ }  // 另外的初始化函数
};
// 问题：忘记调用 init() 怎么办？

// ✅ 构造时就获取
class Good {
public:
    Good() {
        // 构造时就获取资源
        // 要么成功，要么抛异常
    }
};
```

### 实践 3：析构函数不抛异常

```cpp
// ❌ 危险
class Bad {
public:
    ~Bad() {
        if (error) {
            throw std::runtime_error("error");  // 💥 析构函数抛异常！
        }
    }
};
// 如果在栈展开时（异常处理时）再抛异常，程序会直接终止！

// ✅ 安全
class Good {
public:
    ~Good() noexcept {  // 明确标记不抛异常
        try {
            // 清理资源
        } catch (...) {
            // 捕获所有异常，记录日志
        }
    }
};
```

### 实践 4：禁止拷贝或正确实现拷贝

```cpp
// 选项 1：禁止拷贝（推荐）
class NoCopy {
public:
    NoCopy(const NoCopy&) = delete;
    NoCopy& operator=(const NoCopy&) = delete;
};

// 选项 2：深拷贝
class DeepCopy {
public:
    DeepCopy(const DeepCopy& other) {
        // 深拷贝 other 的资源
    }
};
```

---

## 5️⃣ 标准库中的 RAII 类

### 内存管理

```cpp
// unique_ptr：独占所有权
std::unique_ptr<int> p1(new int(10));
std::unique_ptr<int> p2 = std::move(p1);  // 转移所有权

// shared_ptr：共享所有权
std::shared_ptr<int> s1 = std::make_shared<int>(10);
std::shared_ptr<int> s2 = s1;  // 引用计数 +1

// vector：动态数组
std::vector<int> vec(1000);  // 自动管理内存
```

### 文件管理

```cpp
// ifstream/ofstream：文件流
std::ifstream file("data.txt");  // 自动打开
// ... 读取文件 ...
// 自动关闭
```

### 锁管理

```cpp
std::mutex mtx;

// lock_guard：简单锁
{
    std::lock_guard<std::mutex> lock(mtx);  // 加锁
    // ... 临界区 ...
}  // 自动解锁

// unique_lock：灵活锁
{
    std::unique_lock<std::mutex> lock(mtx);  // 加锁
    // 可以手动解锁
    lock.unlock();
    // 可以重新加锁
    lock.lock();
}  // 如果还持有锁，自动解锁

// scoped_lock：多个锁（C++17）
{
    std::scoped_lock lock(mtx1, mtx2);  // 同时锁多个互斥锁
    // ... 临界区 ...
}  // 同时解锁
```

---

## 6️⃣ RAII vs 其他资源管理方式

### 对比：手动管理

```cpp
// 手动管理：容易出错
void manual() {
    int* data = new int[1000];

    if (error1) {
        delete[] data;  // 要记得释放
        return;
    }

    if (error2) {
        delete[] data;  // 又要记得释放
        return;
    }

    delete[] data;  // 正常路径也要释放
}

// RAII：不会出错
void raii() {
    std::vector<int> data(1000);

    if (error1) return;  // 自动释放
    if (error2) return;  // 自动释放

}  // 自动释放
```

### 对比：垃圾回收（GC）

| 特性 | RAII (C++) | GC (Java/Python) |
|------|------------|------------------|
| 释放时机 | **确定**（离开作用域） | **不确定**（GC 运行时） |
| 性能 | **零开销** | **有 GC 停顿** |
| 资源类型 | **所有资源**（内存、文件、锁） | **只有内存** |
| 异常安全 | **保证释放** | **保证释放** |

**RAII 的优势：**
- ✅ 确定性销毁（离开作用域立即释放）
- ✅ 零运行时开销
- ✅ 可以管理任何资源（不只是内存）

---

## 7️⃣ 常见错误

### 错误 1：忘记删除拷贝构造函数

```cpp
// ❌ 危险
class Bad {
public:
    Bad() : data_(new int[1000]) {}
    ~Bad() { delete[] data_; }
    // 没有禁止拷贝！

private:
    int* data_;
};

Bad b1;
Bad b2 = b1;  // 💥 浅拷贝！两个对象指向同一块内存
// b1 析构 → delete[] data_
// b2 析构 → delete[] data_  💥 重复释放！

// ✅ 正确
class Good {
public:
    Good() : data_(new int[1000]) {}
    ~Good() { delete[] data_; }

    // 禁止拷贝
    Good(const Good&) = delete;
    Good& operator=(const Good&) = delete;

private:
    int* data_;
};
```

### 错误 2：资源泄漏

```cpp
// ❌ 危险
class Bad {
public:
    Bad() : data_(new int[1000]) {}
    // 忘记写析构函数！

private:
    int* data_;
};
// 💥 内存泄漏！

// ✅ 正确（更好：不手动管理）
class Good {
public:
    Good() : data_(1000) {}
    // vector 自动管理，不需要写析构函数

private:
    std::vector<int> data_;
};
```

### 错误 3：异常导致资源泄漏

```cpp
// ❌ 危险
void bad() {
    int* data = new int[1000];

    process();  // 可能抛异常

    delete[] data;  // 💥 如果 process() 抛异常，永远不会执行
}

// ✅ 正确
void good() {
    std::vector<int> data(1000);

    process();  // 即使抛异常，vector 也会自动释放
}
```

---

## 8️⃣ 实践练习

### 练习 1：实现一个 Timer 类

要求：
- 构造时开始计时
- 析构时打印耗时

```cpp
#include <chrono>
#include <iostream>

class Timer {
public:
    Timer(const std::string& name) : name_(name) {
        start_ = std::chrono::steady_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << name_ << " took " << duration.count() << "ms\n";
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

// 使用：
void slow_function() {
    Timer timer("slow_function");  // 开始计时

    // ... 耗时操作 ...

}  // 自动打印耗时
```

### 练习 2：实现一个数据库事务类

要求：
- 构造时开始事务
- 析构时提交或回滚

```cpp
class Transaction {
public:
    Transaction(Database& db) : db_(db), committed_(false) {
        db_.begin_transaction();
    }

    ~Transaction() {
        if (!committed_) {
            db_.rollback();  // 如果没有提交，回滚
        }
    }

    void commit() {
        db_.commit();
        committed_ = true;
    }

private:
    Database& db_;
    bool committed_;
};

// 使用：
void transfer_money() {
    Transaction trans(db);  // 开始事务

    db.deduct(account1, 100);
    db.add(account2, 100);

    trans.commit();  // 显式提交
}  // 如果忘记 commit，自动回滚
```

---

## 🎯 总结

### RAII 的核心

1. **构造即获取**：构造函数中获取资源
2. **析构即释放**：析构函数中释放资源
3. **编译器保证**：离开作用域一定调用析构
4. **异常安全**：即使有异常也会释放资源

### RAII 的优势

- ✅ 不会忘记释放资源
- ✅ 异常安全
- ✅ 代码简洁
- ✅ 零运行时开销

### RAII 的规则

1. **Rule of Zero**：尽量用标准库，不自己管理资源
2. **Rule of Five**：如果必须管理资源，定义 5 个特殊函数
3. **禁止拷贝**：大多数 RAII 类应该禁止拷贝
4. **析构不抛异常**：析构函数标记 `noexcept`

---

## 🚀 下一步

学完 RAII 后，接下来学习：
1. **智能指针**（unique_ptr、shared_ptr、weak_ptr）
2. **标准容器**（vector、map、set 等）
3. **移动语义**（深入理解所有权转移）

**配套实践代码：** [practices/01_raii_examples.cpp](practices/01_raii_examples.cpp)
