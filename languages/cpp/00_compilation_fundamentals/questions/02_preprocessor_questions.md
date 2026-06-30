# 预处理器 — 问题版（自测用）

## Q1

`#define PI 3.14` 和 `const double PI = 3.14;` 有什么区别？

<details>
<summary>答案</summary>

| #define | const |
|---|---|
| 预处理期文本替换 | 编译期常量 |
| 没有类型检查 | 有类型检查 |
| 没有作用域 | 有作用域 |
| 调试看不见 | 调试看得见 |

**现代 C++ 能用 const/constexpr 就别用 #define。**

</details>

## Q2

宏的常见坑是什么？如何避免？

<details>
<summary>答案</summary>

```cpp
#define SQUARE(x) x * x
SQUARE(1 + 2)  // 1 + 2 * 1 + 2 = 5，不是 9
```

**每个参数和整个表达式都加括号**：

```cpp
#define SQUARE(x) ((x) * (x))
```

或者直接用 inline 函数代替：

```cpp
inline int square(int x) { return x * x; }
```

</details>

## Q3

条件编译 `#ifdef DEBUG` 一般用来做什么？

<details>
<summary>答案</summary>

```cpp
#ifdef DEBUG
    std::cerr << "x = " << x << "\n";
#endif
```

- 调试模式下打印额外日志，发布模式不编译这些代码
- 跨平台代码（`#ifdef _WIN32` vs `#ifdef __linux__`）
- 特性开关

</details>

## Q4

`#pragma once` 和传统 `#ifndef` header guard 有什么区别？

<details>
<summary>答案</summary>

```cpp
#pragma once          // 一行搞定，不是标准但主流编译器都支持
#ifndef MY_HEADER_H
#define MY_HEADER_H
#endif                // 标准写法，啰嗦但百分之百兼容
```

两者作用相同：防止同一个头文件被多次 include。

</details>

## Q5

预处理阶段能看到的预定义宏有哪些？

<details>
<summary>答案</summary>

```cpp
__FILE__      // 当前文件名
__LINE__      // 当前行号
__DATE__      // 编译日期
__TIME__      // 编译时间
__cplusplus   // 201703L = C++17, 202002L = C++20
```

</details>
