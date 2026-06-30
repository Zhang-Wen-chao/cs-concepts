# 1.3 内存管理 — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

栈（stack）和堆（heap）有什么区别？（至少说 3 点）

<details>
<summary>答案</summary>

| | 栈 | 堆 |
|---|---|---|
| 分配方式 | 自动（编译期决定） | 手动（运行时 new/delete） |
| 速度 | 极快（push/pop） | 慢（查找空闲块） |
| 大小 | 小（MB 级） | 大（GB 级） |
| 生命周期 | 离开作用域自动销毁 | 直到被 delete |
| 用法 | 局部变量、函数参数 | 需要长期存活或大块内存 |

</details>

## Q2

Python 里所有变量都在堆上，C++ 呢？

<details>
<summary>答案</summary>

```cpp
int a = 42;                    // 栈 — 没有 new
std::vector<int> v(1000);      // v 本体在栈，v 里的数据在堆
int* p = new int(42);          // p 在栈，*p 在堆
```

Python 全是堆 — 因为 Python 是动态类型，对象需要 GC 管理。
C++ 值类型默认在栈上，除非你明确 `new` 或者用了动态分配容器。

</details>

## Q3

手写 `new` 和 `delete` 最容易犯的 3 个错误？

<details>
<summary>答案</summary>

1. **忘了 delete** — 内存泄漏。程序跑得越久，吃越多内存
2. **double delete** — delete 了已经 delete 过的指针，**未定义行为**（崩溃 or 数据损坏）
3. **new[] 配 delete 而不是 delete[]** — 只释放了第一个元素，数组其余部分泄漏

```cpp
int* p = new int(42);
delete p;
delete p;           // ❌ 未定义行为

int* arr = new int[10];
delete arr;         // ❌ 应该用 delete[]
```

</details>

## Q4

`malloc/free` 和 `new/delete` 有什么区别？

<details>
<summary>答案</summary>

| | `malloc/free` | `new/delete` |
|---|---|---|
| 语言 | C | C++ |
| 返回值 | `void*`（需要 cast） | 类型安全的指针 |
| 调用构造/析构 | 不调用 | 会调用 |
| 大小 | 手动指定字节数 | 编译器自动 |
| 失败 | 返回 NULL | 抛 `std::bad_alloc` |

`new` ≈ `malloc` + **调用构造函数**，`delete` ≈ **调用析构函数** + `free`

</details>

## Q5

什么是 use-after-free（UAF）？能举个最简单的例子吗？

<details>
<summary>答案</summary>

指针指向的内存已被释放，但还继续用这个指针。

```cpp
int* p = new int(42);
delete p;
*p = 10;  // ❌ use-after-free — 未定义行为（可能崩，可能没崩，都是 bug）
```

Python 有 GC 保证不会 UAF。C++ 你 delete 了就真没了。

</details>

## Q6

为什么说 RAII 可以根治上面这些内存问题？

<details>
<summary>答案</summary>

RAII 把堆内存绑定到栈对象的生命周期：

```cpp
// ❌ 手动管理
int* p = new int(42);
// ... 中间可能 return / 抛异常
delete p;  // 容易忘

// ✅ RAII
std::unique_ptr<int> p = std::make_unique<int>(42);
// 作用域结束自动 delete — 不可能忘，不可能 UAF，不可能 double delete
```

</details>
