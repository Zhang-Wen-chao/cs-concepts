# 预处理深度

> 编译的第一步，发生在编译器真正工作之前。

## 预处理做了什么

```cpp
// 源码
#include <iostream>
#define PI 3.14159
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main() {
    std::cout << PI;
    int x = MAX(10, 20);
}
```

预处理后变成（简化）：

```cpp
// iostream 的全部内容粘贴到这里
// 约 30000 行...

int main() {
    std::cout << 3.14159;
    int x = ((10) > (20) ? (10) : (20));
}
```

**`#include` = 复制粘贴**，不是"导入"或"引用"。

## #define vs const

```cpp
#define PI 3.14159       // 预处理期替换，没有类型检查
const double PI = 3.14159; // 编译期常量，有类型检查
```

**现代 C++ 尽量用 const/constexpr 代替 #define。** 宏没有作用域、没有类型检查、调试困难。

## 宏的坑

```cpp
#define SQUARE(x) x * x
SQUARE(1 + 2)   // 1 + 2 * 1 + 2 = 5，不是 9！
```

正确写法：每个参数和整个表达式都加括号。

```cpp
#define SQUARE(x) ((x) * (x))
```

## 条件编译

```cpp
#ifdef _WIN32
    #include <windows.h>
    #define PLATFORM "Windows"
#elif defined(__linux__)
    #include <unistd.h>
    #define PLATFORM "Linux"
#else
    #define PLATFORM "Unknown"
#endif
```

常用于：跨平台代码、调试模式（`#ifdef DEBUG`）、特性检测。

## 预定义宏

```cpp
__FILE__     // 当前文件名
__LINE__     // 当前行号
__DATE__     // 编译日期
__TIME__     // 编译时间
__cplusplus  // C++ 标准版本（如 201703L = C++17）
```

## 总结

| 预处理指令 | 作用 |
|---|---|
| `#include` | 文件包含（复制粘贴） |
| `#define` | 宏定义（文本替换） |
| `#if`/`#ifdef`/`#endif` | 条件编译 |
| `#pragma once` | 头文件守卫 |
| `#line` | 重置行号（少见） |
| `#error` | 编译时报错 |

**黄金法则**：能不写宏就不写宏。C++ 提供了 const、constexpr、inline、模板等替代方案，比宏安全得多。
