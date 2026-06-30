# 模板特化与偏特化

## 1. 为什么需要特化？

默认模板对大多数类型够用，但某些类型需要特殊处理。比如 `vector<bool>` 要压缩存储。

## 2. 全特化

```cpp
// 通用模板
template <typename T>
struct IsPointer {
    static constexpr bool value = false;
};

// 全特化：针对 T = int*
template <>
struct IsPointer<int*> {
    static constexpr bool value = true;
};

int main() {
    IsPointer<int>::value;     // false
    IsPointer<int*>::value;    // true
}
```

## 3. 偏特化（只有类模板可以）

```cpp
// 通用模板
template <typename T, typename U>
struct Pair {
    static constexpr int category = 0;
};

// 偏特化：两个类型相同时
template <typename T>
struct Pair<T, T> {
    static constexpr int category = 1;
};

// 偏特化：第一个参数是指针时
template <typename T, typename U>
struct Pair<T*, U> {
    static constexpr int category = 2;
};

int main() {
    Pair<int, double>::category;  // 0
    Pair<int, int>::category;     // 1
    Pair<int*, int>::category;    // 2
}
```

**函数模板没有偏特化**。函数想达到类似效果，用**重载**或**标签分发**。

## 4. 实际例子：std::remove_reference

```cpp
template <typename T> struct RemoveRef      { using type = T; };
template <typename T> struct RemoveRef<T&>   { using type = T; };
template <typename T> struct RemoveRef<T&&>  { using type = T; };

RemoveRef<int&>::type;     // int
RemoveRef<int&&>::type;    // int
RemoveRef<int>::type;      // int
```

这就是 `std::remove_reference` 的实现原理。

## 5. 匹配规则

编译器选特化时按 **最特化优先** 原则：

1. 检查所有可用模板
2. 找出匹配的
3. 选参数约束最严格的

```cpp
template <typename T>     void f(T);       // #1 通用
template <typename T>     void f(T*);      // #2 T* 偏特化（函数通过重载模拟）
template <>               void f(int*);    // #3 全特化 int*

f(new int);  // 选 #3
f(new char); // 选 #2
```

## 关键点总结

- **全特化**：`template <>` — 所有参数完全确定，函数和类都能用
- **偏特化**：只固定部分参数 — 仅类模板支持
- 函数模板想实现类似效果用**重载**而非偏特化
- 编译器按**最特化优先**规则匹配特化版本
- 特化是 C++ 模板元编程的基础技巧
