# 特殊成员函数

## 1. 六种特殊成员函数

C++ 编译器在特定条件下会**隐式生成**六种函数：

```cpp
class Thing {
public:
    // 1. 默认构造函数
    Thing();

    // 2. 析构函数
    ~Thing();

    // 3. 拷贝构造函数
    Thing(const Thing&);

    // 4. 拷贝赋值运算符
    Thing& operator=(const Thing&);

    // 5. 移动构造函数    (C++11)
    Thing(Thing&&) noexcept;

    // 6. 移动赋值运算符  (C++11)
    Thing& operator=(Thing&&) noexcept;
};
```

**隐式生成的条件**：你没声明，且所有成员都能对应操作（能拷贝/能移动）。一旦你声明了任何构造/析构/拷贝，部分隐式生成会消失。

## 2. 拷贝 vs 移动

```cpp
class Buffer {
    int* data_;
    size_t size_;
public:
    // 拷贝：深复制
    Buffer(const Buffer& other)
        : data_(new int[other.size_]), size_(other.size_)
    {
        std::copy(other.data_, other.data_ + size_, data_);
    }

    // 移动：偷资源
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;  // 源对象必须置空
        other.size_ = 0;
    }
};
```

| 场景 | 调用 |
|------|------|
| `auto b2 = b1;` | 拷贝构造（b2 是全新副本） |
| `auto b3 = std::move(b1);` | 移动构造（b3 偷走 b1 的资源） |
| `return local_obj;` | 编译器尽量用移动（NRVO/复制消除） |

## 3. Rule of Five

**经典规则**：如果你需要自定义析构函数/拷贝构造/拷贝赋值中的任何一个，说明这个类在管理资源，那么**五个都应该考虑**。

```cpp
class Resource {
    int* ptr_;
public:
    ~Resource();                              // 1: 释放资源
    Resource(const Resource&);                // 2: 深拷贝
    Resource& operator=(const Resource&);     // 3: 拷贝赋值
    Resource(Resource&&) noexcept;            // 4: 移动构造
    Resource& operator=(Resource&&) noexcept; // 5: 移动赋值
};
```

注意：Rule of Five 是从 Rule of Three（C++03，只有构造/拷贝/析构）升级来的。C++11 加了移动，就变成了 Five。

## 4. =default 和 =delete

```cpp
class Safe {
public:
    Safe() = default;                          // 强制生成默认构造
    ~Safe() = default;

    Safe(const Safe&) = delete;                // 禁止拷贝
    Safe& operator=(const Safe&) = delete;

    Safe(Safe&&) = default;                    // 允许移动
    Safe& operator=(Safe&&) = default;
};

// 应用场景：单例、unique_ptr、non-copyable 资源
```

- `= default`：让编译器生成默认版本（比手写空函数更优化）
- `= delete`：显式禁止该函数，任何调用尝试都是编译错误

## 关键点总结

- 编译器会**隐式生成**特殊成员，但规则复杂，建议**显式声明**
- 拷贝 = 深复制，移动 = 偷资源 + 源对象置空
- 移动构造/赋值必须标记 `noexcept`，否则 STL 容器不会选移动
- **Rule of Five**：管理资源的类，五个函数都要考虑
- `= default` 和 `= delete` 显式表达意图，减少 bug
