# 1.2 编译执行 — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

C++ 从源码到可执行文件，是哪四个步骤？

<details>
<summary>答案</summary>

1. **预处理** — 处理 `#include`、`#define`、条件编译
2. **编译** — 源码 → 汇编代码 (`.s`)
3. **汇编** — 汇编代码 → 目标文件 (`.o` / `.obj`)
4. **链接** — 目标文件 + 库 → 可执行文件

</details>

## Q2

`#include <iostream>` 预处理阶段做了什么？

<details>
<summary>答案</summary>

把 `iostream` 头文件的**全部内容**原封不动复制粘贴到当前文件，替换这个 `#include` 行。最终传给编译器的是一大段文本，不是 "引用" 或 "导入"。

所以 `#include` 越多，预处理后的文件越大，编译越慢。

</details>

## Q3

编译阶段（第 2 步）的产出是什么？链接阶段（第 4 步）如果出问题，通常报什么错？

<details>
<summary>答案</summary>

- **编译产出**：汇编代码（`.s` 文件），再汇编成目标文件（`.o`）
- **链接错误典型**：
  - `undefined reference to XXX` — 声明了但没定义（忘了实现 / 忘了链接那个 .o）
  - `multiple definition of XXX` — 同一个符号在多个地方定义了（ODR 违规）

</details>

## Q4

Python 有没有这四个步骤？为什么 C++ 要这么麻烦？

<details>
<summary>答案</summary>

没有。Python 直接**源码 → 字节码 → 解释器执行**，没有独立的编译和链接阶段。

C++ 这么麻烦是因为：
1. **静态类型**需要给所有类型生成对应机器码
2. **零开销原则**— 尽可能在编译期算好，运行时什么都不用干
3. **分离编译**— 改一个源文件只重新编译它自己，不用重新编译整个项目

</details>

## Q5

`.h` 文件里写函数实现（不只是声明），会出什么问题？

<details>
<summary>答案</summary>

如果多个 `.cpp` 文件 `#include` 了同一个 `.h`，链接时会报 **multiple definition**（ODR 违规），因为每个编译单元都有这个函数的定义。

解决：
1. 头文件只写声明，实现在 `.cpp` 里
2. 或者标记 `inline`（允许重复定义，编译器会去重）
3. 或者标记 `static`（每个编译单元有一份独立拷贝）

</details>

## Q6

"只声明不定义" 能编译通过吗？链接呢？

<details>
<summary>答案</summary>

```cpp
// 只声明
extern int x;
void foo();
```

- **编译**能通过 — 编译器相信你在别处定义了
- **链接**通不过 — `undefined reference`，找不到定义

这就是为什么声明放在 `.h`，定义放在 `.cpp`。

</details>
