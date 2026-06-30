# 变参模板 (Variadic Templates)

## 1. 基本语法

C++11 引入的 `...`（参数包），让模板接受任意数量、任意类型的参数。

```cpp
// 声明一个变参函数模板
template <typename... Args>
void print(Args... args);  // Args 是类型包，args 是值包
```

## 2. 递归展开

变参模板最常见的技巧：**编译期递归**。

```cpp
// 基准情况（没有参数时终止递归）
void print_all() {}

// 递归情况：处理第一个，然后展开剩余
template <typename T, typename... Rest>
void print_all(T first, Rest... rest) {
    std::cout << first << ' ';
    print_all(rest...);  // 递归调用，每次吃掉一个参数
}

int main() {
    print_all(1, 3.14, "hello", 'c');  // 输出: 1 3.14 hello c
}
```

## 3. 折叠表达式 (C++17)

C++17 的折叠语法让展开更简洁，无需递归：

```cpp
template <typename... Args>
auto sum(Args... args) {
    return (... + args);       // 左折叠: ((a + b) + c) + d
}

template <typename... Args>
auto sum_right(Args... args) {
    return (args + ...);       // 右折叠: a + (b + (c + d))
}

template <typename... Args>
void print_all(Args... args) {
    (std::cout << ... << args) << '\n';  // 折叠到 cout
}
```

## 4. sizeof... 运算符

```cpp
template <typename... Args>
void count_args(Args... args) {
    std::cout << sizeof...(Args) << " types\n";
    std::cout << sizeof...(args) << " values\n";
}
```

`sizeof...` 在编译期求值，零运行时开销。

## 5. 变参类模板与完美转发

```cpp
// 变参类模板（类似元组实现）
template <typename...> class Tuple;

// 递归定义
template <typename T, typename... Rest>
class Tuple<T, Rest...> {
    T head_;
    Tuple<Rest...> tail_;
};

// 完美转发参数包
template <typename... Args>
auto make_unique_wrapper(Args&&... args) {
    return std::make_unique<std::tuple<Args...>>(
        std::forward<Args>(args)...
    );
}
```

`std::forward<Args>(args)...` 是变参完美转发的标准模式。

## 关键点总结

- `...` 定义参数包，使用时展开
- 经典展开方式：**递归**（C++11/14）或 **折叠表达式**（C++17）
- `sizeof...(Args)` 在编译期获取参数数量
- 变参 + 完美转发 = 工厂函数/包装器的标配
- STL 中的 `tuple`, `variant`, `make_shared` 都靠变参模板实现
