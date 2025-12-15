# 模板基础

## 核心思想

**一次编写，处处复用。编译器为每种类型生成代码。**

## 函数模板

```cpp
// 定义
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// 使用（自动推导类型）
max(3, 5);           // T = int
max(1.5, 2.5);       // T = double

// 显式指定类型
max<int>(3, 5);
```

## 类模板

```cpp
// 定义
template<typename T>
class Box {
    T value_;
public:
    Box(T v) : value_(v) {}
    T get() const { return value_; }
};

// 使用（必须显式指定类型）
Box<int> b1(42);
Box<std::string> b2("hello");
```

## 多个模板参数

```cpp
template<typename T, typename U>
auto add(T a, U b) {
    return a + b;
}

add(3, 1.5);  // T=int, U=double, 返回 double
```

## 常见陷阱

```cpp
// ❌ 模板定义必须在头文件
// foo.cpp
template<typename T>
T add(T a, T b) { return a + b; }  // 链接错误

// ✅ 模板定义放头文件
// foo.h
template<typename T>
T add(T a, T b) { return a + b; }

// ❌ 类型推导失败
template<typename T>
T max(T a, T b) { return a > b ? a : b; }
max(3, 1.5);  // 错误：T 既是 int 又是 double

// ✅ 显式指定或用 auto
max<double>(3, 1.5);
template<typename T, typename U>
auto max(T a, U b) { return a > b ? a : b; }
```

## 实用示例

```cpp
// 通用打印
template<typename T>
void print(const T& value) {
    std::cout << value << std::endl;
}

// 容器打印
template<typename Container>
void print_all(const Container& c) {
    for (const auto& item : c) {
        std::cout << item << " ";
    }
}
```

## 要点

1. **函数模板自动推导类型，类模板需显式指定**
2. **模板定义必须在头文件**
3. **标准库大量使用模板**（vector、map 等）
4. **简单场景用模板，复杂场景考虑其他方案**
