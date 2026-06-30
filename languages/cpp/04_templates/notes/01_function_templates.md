# 函数模板

## 1. 为什么需要模板？

函数重载能处理不同类型的参数，但每种类型都要写一遍。模板让编译器**根据调用自动生成**对应版本。

```cpp
// 一个模板 = 编译器自动生成多个重载
template <typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    max(3, 5);        // 实例化为 max<int>(int, int)
    max(3.14, 2.72);  // 实例化为 max<double>(double, double)
    // max(3, 4.2);   // ❌ 模板参数推导冲突：int vs double
}
```

## 2. 模板参数推导

编译器根据调用参数推导模板参数，推导规则和函数重载一样。

```cpp
template <typename T>
void foo(T a, T b) {}

template <typename T>
void bar(T a, const T& b) {}

// 显式指定模板参数
auto r = max<double>(3, 4.2);  // 强制 double 版本

// 多个模板参数
template <typename R, typename T, typename U>
R convert(T t, U u) { return static_cast<R>(t + u); }
```

## 3. 模板特化

```cpp
// 基础模板
template <typename T>
bool equal(T a, T b) { return a == b; }

// 全特化：针对 const char*
template <>
bool equal<const char*>(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}
```

## 4. 简化的模板语法 (C++20)

```cpp
// C++20 可以用 auto 做模板参数
auto max(auto a, auto b) { return a > b ? a : b; }
// 等价于 template <typename T> T max(T a, T b)
```

## 5. 编译期多态

模板是 C++ **编译期多态**的手段，和虚函数的运行时多态互补：

| 模板（静态多态） | 虚函数（动态多态） |
|------------------|--------------------|
| 编译期决定 | 运行时决定 |
| 可内联 | 不可内联 |
| 每套类型产生一个实例（代码膨胀） | 共享一份代码 |
| 更灵活（所有类型都能用） | 更约束（必须是同一继承体系） |

## 关键点总结

- 模板 = 代码生成器，在编译期按需实例化
- 模板参数可以**隐式推导**或**显式指定**
- 全特化可以覆盖特定类型的默认实现
- 模板是**编译期多态**，虚函数是**运行时多态**
- 显式指定模板参数可解决类型推导冲突
