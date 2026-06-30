# 1.9 未定义行为 (UB) — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

什么是未定义行为（Undefined Behavior / UB）？一句话解释。

<details>
<summary>答案</summary>

**C++ 标准说"我不负责"的行为**。编译器可以做出任何反应：正常工作、崩、删除你的文件（理论上）、让你的代码产生随机的错误结果。

UB 是 C++ 最大的坑，也是和 Python 的最大区别之一。

</details>

## Q2

列出至少 5 种常见的未定义行为。

<details>
<summary>答案</summary>

1. **use-after-free** — 用了已 delete 的指针
2. **double delete** — 对同一个指针 delete 两次
3. **数组越界** — `int arr[5]; arr[100] = 42;`
4. **整数溢出** — `int x = INT_MAX; x++;`（有符号）
5. **空指针解引用** — `int* p = nullptr; *p = 42;`
6. **除以零** — `int x = 1 / 0;`
7. **未初始化的变量** — `int x; cout << x;`
8. **dangling reference** — 返回局部变量的引用

</details>

## Q3

数组越界在 Python 和 C++ 里分别怎么表现？

<details>
<summary>答案</summary>

```python
# Python — 友好
arr = [1, 2, 3]
arr[100] = 42  # ❌ IndexError: list index out of range
```

```cpp
// C++ — 危险
int arr[3] = {1, 2, 3};
arr[100] = 42;  // ❌ 未定义行为 — 可能崩，可能改到了别的变量，可能看起来正常
```

Python 是**下标检查** + **友好报错**。C++ `[]` 运算符直接算偏移地址，不检查边界（为了性能），越界就是 UB。

</details>

## Q4

有符号整数溢出是 UB，无符号整数溢出呢？

<details>
<summary>答案</summary>

```cpp
int x = INT_MAX; x++;        // ❌ 有符号溢出 = UB
unsigned y = UINT_MAX; y++;  // ✅ 无符号溢出 = 回绕到 0（标准定义行为）
```

C++ 标准说无符号整数行为是"模 2^n"算术，不会 UB。有符号整数必须是二的补码表示，但溢出依然是 UB，**编译器可能假设"有符号整数不会溢出"来做优化**。

</details>

## Q5

为什么"看起来能跑"是 C++ 代码最危险的状态？

<details>
<summary>答案</summary>

UB 的可怕之处在于它**不确定**。同样的代码，今天跑对明天跑崩，g++ 编译正常 clang 编译崩，开优化崩溃不开优化正常。

最坑的是：UB 可能**没有即时表现** — 越界写了一段内存，当时函数返回了，但 10 分钟后另一个函数因为那段内存被改了而崩溃。你要定位到真实原因很难。

所以：**开了 sanitizer 编译跑过 ≠ 没 UB**。但不开 sanitizer ≈ 裸奔。

</details>

## Q6

怎么尽可能避免未定义行为？

<details>
<summary>答案</summary>

1. **永远开 sanitizer** 编译调试版：`-fsanitize=address,undefined`
2. **坚持用 RAII 和智能指针** — 自动管理资源，不手写 new/delete
3. **用 `std::array` 和 `std::vector::at()`** 代替裸数组（`at()` 有越界检查）
4. **编译器 warning 全开**：`-Wall -Wextra -Wpedantic`，warning 当 error 修
5. **初始化所有变量**：`int x = 0;` 别只写 `int x;`
6. **代码 review** 注意所有"看起来危险"的地方

</details>
