# 1.8 头文件 & ODR — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

声明（declaration）和定义（definition）的区别是什么？

<details>
<summary>答案</summary>

- **声明**：告诉编译器"有这个东西"，但不分配存储空间
- **定义**：实际创建/实现它，分配存储或生成代码

```cpp
extern int x;       // 声明 — x 在别处定义
int x = 42;         // 定义 — x 就在这里

void foo();         // 声明
void foo() {}       // 定义
```

声明可以出现多次，定义在整个程序里只能有一次（ODR）。

</details>

## Q2

ODR 是什么的缩写？核心规则是什么？

<details>
<summary>答案</summary>

**One Definition Rule** — 一次定义原则。

核心规则：在每个翻译单元（每个 `.cpp`）里，每个函数、变量、类、模板只能有一个定义。整个程序里，非内联的函数和全局变量也只能有一个定义。

违反就报：`multiple definition of XXX` 链接错误。

</details>

## Q3

头文件守卫（header guard）写法有几种？不写会怎样？

<details>
<summary>答案</summary>

两种写法：

```cpp
// 1. #pragma once — 简单，但不是 C++ 标准（但所有主流编译器都支持）
#pragma once

// 2. 传统 include guard
#ifndef MY_HEADER_H
#define MY_HEADER_H
// ...
#endif
```

不写头文件守卫 → 一个 `.cpp` 里两个 `#include` 导入了同一个头文件 → 类型/函数重复定义 → 编译错误。

</details>

## Q4

`#include "my_header.h"` 和 `#include <iostream>` 的查找路径有什么区别？

<details>
<summary>答案</summary>

- `""` : 先从当前源文件所在目录找，找不到再去系统 include 路径找
- `<>` : 直接从系统 include 路径找（`-I` 参数指定的也算）

所以自己的头文件用 `""`，标准库用 `<>`。

</details>

## Q5

为什么 `.h` 文件里可以写类定义和 inline 函数，但不能写普通的函数定义？

<details>
<summary>答案</summary>

- **类定义**每个编译单元都需要知道类布局才能创建对象，且 ODR 允许重复的类定义（只要内容一致）
- **inline 函数**允许重复定义（编译器会去重）
- **普通函数定义**如果在 `.h` 里，被多个 `.cpp` include 了 → 链接时报 multiple definition

</details>

## Q6

"翻译单元"（translation unit）是什么？

<details>
<summary>答案</summary>

一个 `.cpp` 文件 + 它 `#include` 的所有头文件展开后的整体。预处理完成后，编译器编译的最小单位就是一个翻译单元。

你改了一个 `.h` 文件，所有包含它的翻译单元都得重新编译。这就是为什么大型项目编译慢。

</details>
