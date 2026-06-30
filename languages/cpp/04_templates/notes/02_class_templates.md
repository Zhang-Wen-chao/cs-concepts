# 类模板

## 1. 基本语法

```cpp
template <typename T>
class Box {
    T value_;
public:
    explicit Box(T v) : value_(v) {}

    T get() const { return value_; }
    void set(T v) { value_ = v; }
};

// 使用
Box<int> intBox(42);
Box<std::string> strBox("hello");
```

## 2. 成员函数定义

类模板的成员函数同样是模板，定义时也必须带 `template`。

```cpp
template <typename T>
class Stack {
    std::vector<T> data_;
public:
    void push(const T& v);
    T pop();
};

// 类外定义成员函数
template <typename T>
void Stack<T>::push(const T& v) {
    data_.push_back(v);
}

template <typename T>
T Stack<T>::pop() {
    T v = std::move(data_.back());
    data_.pop_back();
    return v;
}
```

## 3. 模板嵌套

```cpp
// 嵌套模板参数
template <typename T, typename Container = std::vector<T>>
class Queue {
    Container data_;
public:
    void push(const T& v) { data_.push_back(v); }
};
```

## 4. 模板与友元

```cpp
template <typename T>
class Point {
    T x_, y_;
public:
    // 每个 Point<T> 实例有自己的友元函数
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        return os << '(' << p.x_ << ',' << p.y_ << ')';
    }
};
```

## 5. typename 与 class

```cpp
template <typename T>  // typename 和 class 完全等价
template <class T>     // 但在嵌套依赖类型时必须用 typename
```

```cpp
template <typename T>
void foo() {
    // 如果 T::value_type 是类型，必须加 typename 告诉编译器
    typename T::value_type v;
}
```

## 关键点总结

- 类模板是类型生成器：`Box<int>` 和 `Box<double>` 是完全不同的类型
- 成员函数在类外定义时必须重复 `template <typename T>`
- 模板参数可以有**默认值**（如 Container 默认是 vector）
- 在模板中引用依赖类型时用 `typename` 消歧
- STL 容器（vector, map 等）都是类模板
