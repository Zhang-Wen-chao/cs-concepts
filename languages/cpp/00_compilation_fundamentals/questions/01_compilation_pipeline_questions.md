# 编译管线 — 问题版（自测用）

## Q1

C++ 从源码到可执行文件是哪四个步骤？

<details>
<summary>答案</summary>

1. **预处理** — `#include` / `#define` / 条件编译
2. **编译** — 源码 → 汇编代码 `.s`
3. **汇编** — 汇编代码 → 目标文件 `.o`
4. **链接** — 目标文件 + 库 → 可执行文件

</details>

## Q2

预处理阶段 `#include <iostream>` 具体做了什么？

<details>
<summary>答案</summary>

把 `iostream` 头文件的内容原封不动复制粘贴，替换这一行 `#include`。最终给编译器的是一个巨大的文本文件，不是"导入"或"引用"。

</details>

## Q3

编译阶段如果只声明不实现一个函数，能通过编译吗？链接呢？

<details>
<summary>答案</summary>

- **编译**：能通过。编译器相信你在别处定义了
- **链接**：通不过，报 `undefined reference`

这就是为什么声明放 `.h`，定义放 `.cpp`。

</details>

## Q4

翻译单元（translation unit）是什么？

<details>
<summary>答案</summary>

一个 `.cpp` 文件 + 它 `#include` 的所有头文件展开后的整体。编译器最小编译单位。

你改了一个 `.h`，所有包含它的翻译单元都要重新编译。

</details>

## Q5

"改一个源文件只重新编译它自己" 是 C++ 哪个阶段的优势？

<details>
<summary>答案</summary>

**分离编译** — 每个 `.cpp` 独立编译成 `.o`，链接时只链接改了的部分。大规模项目不需要每次全量编译。

对比：Python 没有这个阶段，每次都是从头解析执行。

</details>
