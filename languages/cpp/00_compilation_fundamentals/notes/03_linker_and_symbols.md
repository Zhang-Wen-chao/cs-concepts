# 链接器与符号

> 每个 .cpp 单独编译，链接器把它们粘在一起。

## 符号（Symbol）

每个函数和全局变量在目标文件中对应一个**符号**：

```cpp
// foo.cpp
int global_var = 42;           // 符号: _global_var (数据)
void foo() {}                  // 符号: _foo (代码)

// main.cpp
extern int global_var;         // 引用符号 _global_var
void foo();                    // 引用符号 _foo
int main() { return 0; }      // 符号: _main
```

## 常见链接错误

### 1. undefined reference

```cpp
// 声明了但没定义
extern int x;
int main() { return x; }
// 链接: undefined reference to `x'
```

声明 = 承诺"别的地方有定义"。链接器找不到定义就报错。

### 2. multiple definition

```cpp
// 同一个符号在多个 .cpp 定义了
// a.cpp: int foo() { return 1; }
// b.cpp: int foo() { return 2; }
// 链接: multiple definition of `foo'
```

## 编译模型

```
源码 ──预处理──→ 翻译单元 ──编译──→ 目标文件 ──链接──→ 可执行文件
.cp  [#include]  [巨大文本]   [-c]    .o      [ld]   a.out
```

每个 `.o` 文件里可能有**未解析的符号**（调用了别的文件定义的函数）。链接器的工作就是把这些符号补齐。

## 声明 vs 定义回顾

| 代码 | 是声明？ | 是定义？ |
|------|:---:|:---:|
| `extern int x;` | ✅ | ❌ |
| `int x;` (全局) | ✅ | ✅ |
| `int x = 42;` | ✅ | ✅ |
| `void foo();` | ✅ | ❌ |
| `void foo() {}` | ✅ | ✅ |
| `class X { };` | ✅ | ✅ |
| `struct X;` | ✅ | ❌（向前声明） |

## 头文件的最佳实践

```cpp
// example.h
#pragma once

void func();                    // 函数声明放头文件
extern int global;              // extern 声明放头文件

// example.cpp
#include "example.h"
void func() { /* 定义 */ }     // 实现放 .cpp
int global = 42;                // 定义放 .cpp
```

## 总结

- 编译是**各个击破**，链接是**合而为一**
- `.h` 放声明，`.cpp` 放定义
- `undefined reference` = 有声明没定义
- `multiple definition` = 定义被重复编译了
