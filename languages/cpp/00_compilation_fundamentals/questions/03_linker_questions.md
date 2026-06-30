# 链接器与符号 — 问题版（自测用）

## Q1

"声明"和"定义"有什么区别？

<details>
<summary>答案</summary>

- **声明**：告诉编译器"有这么个东西，别报错"，不分配存储
- **定义**：实际创建，分配存储或生成代码

```cpp
extern int x;    // 声明
int x = 42;      // 定义
void foo();      // 声明
void foo() {}    // 定义
```

声明可以出现多次，定义只能一次（ODR — One Definition Rule / 一次定义原则）。

</details>

## Q2

`undefined reference to 'foo'` 是什么情况下报的？怎么修？

<details>
<summary>答案</summary>

链接时报的。原因：声明了 `foo`，编译器信了，链接器找不到 `foo` 的定义。

修法：
1. 实现了 `foo` 但忘记编译它所在 `.cpp` → 加进编译命令
2. 实现了但忘记链接对应的 `.o` → 加上
3. 用了外部库但没链接 → 加 `-lxxx`

</details>

## Q3

`multiple definition of 'foo'` 是什么情况下报的？

<details>
<summary>答案</summary>

同一个函数在多个翻译单元中都有定义。常见原因：

1. 头文件里写了函数实现，被多个 `.cpp` include
2. 两个 `.cpp` 里各写了一个同名函数

修法：头文件只写声明，实现在 `.cpp`；或者用 `inline` 标记。

</details>

## Q4

`extern int x;` 和 `int x;`（在全局）有什么区别？

<details>
<summary>答案</summary>

- `extern int x;` — 声明，承诺"x 在某个 `.cpp` 里定义了"
- `int x;`（全局）— 定义，分配了存储空间（暂定定义 / tentative definition）

`extern` 本身就是"声明"标志，不分配存储。

</details>

## Q5

ODR 的核心规则是什么？

<details>
<summary>答案</summary>

每个**非 inline** 的函数、变量、全局对象在整个程序中**只能有一个定义**。

例外：
- `inline` 函数可以有多份定义（编译器会去重）
- 类定义可以重复（只要内容一致）
- 模板实例化可以重复

</details>
