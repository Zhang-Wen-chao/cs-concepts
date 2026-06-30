# Stage 0: 编译底层 — 代码怎么变成可执行文件

> 面试高频问题：include、声明/定义、链接错误、static 都从这里出

## 编译四步

```
hello.cpp
   │
   │ 1. 预处理 (Preprocessing)        ← 处理 #include, #define, 条件编译
   ▼
hello.i   (纯 C++，展开所有头文件)
   │
   │ 2. 编译 (Compilation)            ← .cpp → 汇编代码
   ▼
hello.s   (汇编文件)
   │
   │ 3. 汇编 (Assembly)               ← 汇编 → 机器码
   ▼
hello.o   (目标文件，ELF/Mach-O 格式)
   │
   │ 4. 链接 (Linking)                 ← 多个 .o + 库文件 → 可执行文件
   ▼
hello     (可执行文件)
```

## 自己跑一遍看到每个产物

```bash
mkdir -p /tmp/cpp_stage0 && cd /tmp/cpp_stage0

cat > hello.cpp << 'EOF'
#include <iostream>
#define GREETING "Hello"

int main() {
    std::cout << GREETING << ", C++!" << std::endl;
    return 0;
}
EOF

# 1. 预处理：看宏展开后的样子（会非常长）
clang++ -E hello.cpp -o hello.i
head -20 hello.i         # 你会看到 iostream 的全部内容被展开了

# 2. 编译：变成汇编
clang++ -S hello.cpp -o hello.s
cat hello.s              # 看到 _main 标签和具体指令（mov, call 等）

# 3. 汇编：变成机器码（目标文件）
clang++ -c hello.cpp -o hello.o
file hello.o             # Mach-O 64-bit object

# 4. 链接：变成可执行文件
clang++ hello.o -o hello
./hello                  # 输出: Hello, C++!
```

## 关键概念

### 1. 声明 vs 定义

```cpp
// 声明 (declaration)：告诉编译器"这个符号存在"
int add(int a, int b);
extern int global_var;
void foo();

// 定义 (definition)：分配存储空间 / 提供实现
int add(int a, int b) { return a + b; }   // 函数定义
int global_var = 42;                       // 变量定义（分配空间）
```

**C++ 单一定义规则 (ODR)**：非内联函数/非模板的普通函数在整个程序中**只能有一个定义**。声明可以多次。

### 2. 头文件的职责

头文件（`.h` / `.hpp`）放**声明**，源文件（`.cpp`）放**定义**。

```cpp
// math.h
int add(int a, int b);           // 声明

// math.cpp
#include "math.h"
int add(int a, int b) {          // 定义
    return a + b;
}

// main.cpp
#include "math.h"
int main() {
    return add(1, 2);            // 编译器只需要看到声明
}
```

**头文件保护**（防止重复包含）：
```cpp
// my_header.h
#pragma once                 // 简单方式：现代编译器都支持
// 或者传统方式：
#ifndef MY_HEADER_H
#define MY_HEADER_H
// ... 内容 ...
#endif
```

### 3. 编译错误 vs 链接错误

| 错误类型 | 阶段 | 表现 | 例子 |
|---|---|---|---|
| 编译错误 | 单个 `.cpp` 编译时 | 符号没生成 | 语法错、类型不匹配、找不到声明 |
| 链接错误 | 所有 `.cpp` 都过了 | 找不到符号定义 | `undefined reference to 'add'` |

例：只声明不定义 → 编译过，链接挂：
```cpp
// link_err.cpp
int add(int a, int b);   // 只声明
int main() { return add(1,2); }
```
```bash
clang++ link_err.cpp -o link_err
# linker error: undefined reference to 'add(int, int)'
```

### 4. static 关键字在不同位置

**面试高频陷阱题**：

```cpp
// (a) 函数内 static：静态局部变量，只初始化一次
void counter() {
    static int n = 0;   // 整个程序生命周期
    ++n;
    std::cout << n;
}

// (b) 文件作用域 static：内部链接，只在本文件可见
static int helper = 42;  // 其他 .cpp 看不到 helper

// (c) 类内 static：所有对象共享
class Foo {
    static int count;    // 类的，不是对象的
};
```

**内部链接 vs 外部链接**：
- 默认全局变量/函数是**外部链接**（其他 `.cpp` 能看到）
- 加 `static` 变**内部链接**（只在本文件）
- `extern` 显式声明外部链接

## 看符号表

```bash
# 看 .o 里有什么符号（重要技能，面试调试用）
nm hello.o | head
# 或者更详细
nm -C hello.o            # -C demangle C++ 名字
```

你会看到：
```
0000000000000000 T _main
```

`T` = 在 text 段（代码段）定义的全局符号。

C++ 函数名会被**名字修饰 (name mangling)**：
```bash
nm -C hello.o
# 可能看到: _Z3addii   (原始: add(int, int))
# 解开就是 add(int, int)
```

这就是为什么 C++ 能支持函数重载——编译器把参数类型编码进符号名。

## 实战任务

`practices/00_compilation/` 下完成：
1. 跑一遍上面的四步命令，看每个产物
2. 故意制造一个编译错误、一个链接错误，记录报错
3. 用 `nm` 看 `.o` 文件的符号表
4. 写一个两文件项目（math.h + math.cpp + main.cpp），手动 `clang++` 链接

## 自测问题（能讲清就算过）

1. `#include <iostream>` 在预处理后变成了什么？
2. 编译错误和链接错误有什么区别？分别在哪个阶段发现？
3. 头文件为什么要加 `#pragma once` 或 `#ifndef` 保护？
4. C++ 为什么要做 name mangling？跟函数重载什么关系？
5. `static` 在函数内、文件作用域、类内三种位置分别是什么含义？

## 下一步

环境熟悉 + 四步跑通后，进入 **Stage 1: C++ vs Python 本质差异**，开始写真正的 C++ 代码。
