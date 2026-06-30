# 类与对象基础

## 1. 类是什么？

C++ 的 `class` 是用户自定义类型，把 **数据** 和 **操作** 打包在一起。编译器不主动生成任何东西，你写多大，对象就占多大（除了虚函数表的隐式指针，后面讲）。

```cpp
class Point {
public:
    // Constructor: 创建对象时自动调用
    Point(int x, int y) : x_(x), y_(y) {}  // 成员初始化列表

    // Member function
    int area() const { return x_ * y_; }    // const 承诺不改成员

private:
    int x_, y_;  // 默认 private，外部不能直接访问
};
```

## 2. 访问控制

| 关键字 | 含义 |
|--------|------|
| `public` | 谁都能访问 |
| `protected` | 自己 + 派生类能访问 |
| `private` | 只有自己（和 friend）能访问 |

C++ 的访问控制是 **编译期** 的，不增加运行时开销。

## 3. 构造函数与成员初始化列表

成员初始化列表是 C++ 的特有语法，**必须在构造函数体执行前完成**。

```cpp
class Widget {
    const int id_;          // const 成员必须在初始化列表初始化
    std::string name_;
    std::vector<int> data_;

public:
    // 初始化列表按成员声明顺序执行，不是列表书写顺序！
    Widget(int id, std::string name)
        : id_(id)
        , name_(std::move(name))  // 移动而非拷贝
        , data_(100, 0)           // 直接构造，避免先默认构造再赋值
    {}
};
```

**为什么用初始化列表？** 不用的话，成员会先默认构造再赋值，多一次无意义操作。

## 4. 析构函数

```cpp
class Buffer {
    int* ptr_;
    size_t size_;
public:
    Buffer(size_t n) : ptr_(new int[n]), size_(n) {}
    // ~ 开头，无参数无返回值，不能重载
    ~Buffer() { delete[] ptr_; }
};
```

**RAII 的精髓**：资源在构造时获取，析构时释放。栈上对象离开作用域自动析构。

## 5. this 指针

每个非静态成员函数都有一个隐式参数 `this`，指向调用该函数的对象。

```cpp
class Foo {
    int x;
public:
    void set(int x) {
        this->x = x;  // 必须用 this 区分参数和成员
    }
    Foo& chain(int x) { this->x = x; return *this; }  // 链式调用
};
```

## 关键点总结

- **struct vs class**：struct 默认 public，class 默认 private，其他完全一样
- **初始化列表** 按成员 **声明顺序** 执行，不是列表顺序
- **const 成员** 和 **引用成员** 必须在初始化列表初始化
- **析构顺序** 和构造顺序相反（先构造的后析构）
- **RAII** 是 C++ 资源管理的基石
