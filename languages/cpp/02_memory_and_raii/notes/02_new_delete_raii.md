# new/delete 与 RAII

## new 和 delete 怎么工作

```cpp
// new = 分配内存 + 调用构造函数
int* p = new int(42);         // 分配 4 字节 + int(42) 构造
int* arr = new int[10];       // 分配 40 字节 + 默认构造每个元素

// delete = 调用析构函数 + 释放内存
delete p;                     // 析构 + free
delete[] arr;                 // 逐个析构 + free
```

**`new[]` 必须配 `delete[]`**，否则只有第一个元素被析构。

## 手动管理的痛苦

```cpp
void processFile(const char* filename) {
    char* buffer = new char[1024];
    FILE* f = fopen(filename, "r");

    if (!f) {
        delete[] buffer;     // 忘了写 = 泄漏
        return;
    }

    if (some_condition()) {
        fclose(f);
        delete[] buffer;     // 忘了写 = 泄漏
        return;              // 每一路返回都得写 cleanup
    }

    // ... 处理
    fclose(f);
    delete[] buffer;
}
```

**问题**：太多出口、容易漏、异常不安全。

## RAII 解决

RAII = Resource Acquisition Is Initialization。**把资源的生命周期绑定到对象的生命周期。**

```cpp
class Buffer {
    char* data_;
public:
    explicit Buffer(size_t size) : data_(new char[size]) {}
    ~Buffer() { delete[] data_; }

    // 禁止拷贝（否则两个对象指向同一块内存，double delete）
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    char* get() { return data_; }
};

void processFile(const char* filename) {
    Buffer buf(1024);          // 获取资源
    FILE* f = fopen(filename, "r");
    if (!f) return;            // buf 自动释放 ✅
    if (some_condition()) return; // buf 自动释放 ✅
    fclose(f);
}                              // buf 自动释放 ✅
```

## 智能指针 = RAII 容器

```cpp
std::unique_ptr<int> p = std::make_unique<int>(42);
std::shared_ptr<int> q = std::make_shared<int>(42);
```

这些就是 RAII 封装好了的工业级版本。自己写的 `IntPtr` / `Buffer` 不用在生产用，用标准库的。

## 总结

| 概念 | 含义 |
|---|---|
| RAII | 构造获取资源，析构释放资源 |
| 手动管理 | new/delete 在显式位置，容易泄漏 |
| 智能指针 | unique_ptr / shared_ptr，工业级 RAII |
| 异常安全 | RAII 在异常栈展开时自动释放 |
