# 移动语义

> 避免不必要的拷贝，C++11 最重要的特性。

## 问题

```cpp
std::vector<int> createBigVector() {
    std::vector<int> v(1000000);
    return v;     // C++11 前：拷贝 100 万个 int → 慢
}
```

C++11 前，返回一个大对象就是拷贝所有数据。C++11 后：**移动**。

## 左值和右值

```cpp
int x = 42;    // x 是左值（有名字，可取地址）
int y = x;     // x 是左值
int z = 42;    // 42 是右值（没有名字，临时量）
```

- **左值**：可以出现在赋值号左边。有地址。
- **右值**：只能出现在右边。临时量，用完就销毁。

## std::move

```cpp
std::string a = "hello";
std::string b = std::move(a);  // 把 a 变成右值，触发移动构造

std::cout << a;  // ✅ 有效但未指定状态（通常是空字符串）
```

**`std::move` 没有移动任何数据。** 它只是把参数强制转型为右值引用（`T&&`），让编译器选择移动构造函数而不是拷贝构造函数。

## 移动构造函数

```cpp
class Buffer {
    char* data_;
    size_t size_;
public:
    // 移动构造 — 偷资源，不是拷贝
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;   // 让源对象不再持有
        other.size_ = 0;
    }

    // 拷贝构造（禁止）
    Buffer(const Buffer&) = delete;
};
```

移动 ≈ **转移所有权**。拷贝 ≈ **复制数据**。

## 什么时候自动移动

```cpp
// 1. 函数返回局部变量
std::vector<int> makeVec() {
    std::vector<int> v;
    return v;     // 自动移动（或 NRVO 省略）

    // 2. 传值参数构造
    void Widget::setName(std::string name) {
        name_ = std::move(name);
    }

    // 3. 临时量传参
    vec.push_back(std::string("hello"));  // 移动构造
}
```

## Rule of Five

如果你的类需要自定义析构函数、拷贝构造、拷贝赋值中的任何一个，那么**通常五个都需要**（C++11 后）：

| 特殊成员函数 | 作用 |
|---|---|
| 析构函数 | 释放资源 |
| 拷贝构造函数 | 深拷贝资源 |
| 拷贝赋值运算符 | 深拷贝资源 |
| 移动构造函数 | 转移资源所有权 |
| 移动赋值运算符 | 转移资源所有权 |

```cpp
class Resource {
    int* data_;
public:
    ~Resource() { delete data_; }

    Resource(const Resource& other) : data_(new int(*other.data_)) {}
    Resource& operator=(const Resource& other) {
        if (this != &other) {
            delete data_;
            data_ = new int(*other.data_);
        }
        return *this;
    }

    Resource(Resource&& other) noexcept : data_(other.data_) {
        other.data_ = nullptr;
    }
    Resource& operator=(Resource&& other) noexcept {
        if (this != &other) {
            delete data_;
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }
};
```

## 总结

| 操作 | 代价 | 源对象状态 |
|---|---|---|
| 拷贝 | 分配 + 复制 | 不变 |
| 移动 | 交换指针 | 有效但未指定 |
| 省略（NRVO） | 零 | N/A |

**黄金法则**：如果不需要自定义析构/拷贝，用 `= default`。编译器生成的移动语义通常够用。
