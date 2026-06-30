# 1.6 RAII — 问题版（自测用）

> 盖住答案，能说上来就算过关。说不上来 → 说明上次白讲了，这次好好学。

## Q1

RAII 的全称是什么？翻译成中文。

<details>
<summary>答案</summary>

**Resource Acquisition Is Initialization** — 资源获取即初始化。

</details>

## Q2

RAII 的核心思想是什么？（一句话）

<details>
<summary>答案</summary>

**把资源的生命周期绑定到对象的生命周期**。构造时获取资源，析构时自动释放。

</details>

## Q3

以下哪些是"资源"？（多选）
A. 堆内存
B. 文件句柄
C. 互斥锁
D. 网络 socket
E. 数据库连接

<details>
<summary>答案</summary>

**全部都是**。任何「用完后必须归还/关闭/释放」的东西都是资源。

</details>

## Q4

Python 有没有 RAII？举一个例子。

<details>
<summary>答案</summary>

有。`with open("file.txt") as f:` — 进入 with 块获取资源（打开文件），退出 with 块自动释放（关闭文件）。Python 叫 **上下文管理器**，C++ 叫 **RAII**。

</details>

## Q5

C++ 里 RAII 靠什么机制自动触发？

<details>
<summary>答案</summary>

**析构函数** (`~ClassName()`)。对象生命周期结束（离开作用域 / 被 delete / 异常栈展开）时，编译器自动调用析构函数。

</details>

## Q6

手写一个最简单的 RAII 类封装，管理一个 `int*` 堆内存。

<details>
<summary>答案</summary>

```cpp
class IntPtr {
    int* ptr_;
public:
    IntPtr(int val) : ptr_(new int(val)) {}       // 构造: 获取资源
    ~IntPtr() { delete ptr_; }                     // 析构: 释放资源

    int get() const { return *ptr_; }
};

// 使用
{
    IntPtr p(42);        // new int(42)
    cout << p.get();     // 42
}                        // 自动 delete — 不会泄漏
```

</details>

## Q7

如果不写 RAII，手动 `new` / `delete` 会有什么问题？

<details>
<summary>答案</summary>

1. **忘了 delete** → 内存泄漏
2. **中间 return 或抛异常** → 跳过 delete
3. **多路条件分支** → 每个出口都要写 delete，容易漏
4. **double delete** → 未定义行为

RAII 让析构函数自动处理，以上全避免。

</details>

## Q8

`unique_ptr` 是不是 RAII？`shared_ptr` 呢？为什么？

<details>
<summary>答案</summary>

都是。它们都在构造函数中获取内存资源，析构函数中释放。`unique_ptr` 用 delete，`shared_ptr` 用引用计数控制块 + 最终 delete。这就是智能指针用 RAII 包装了裸指针。

</details>

## Q9

C++ 标准库里还有哪些 RAII 的例子？

<details>
<summary>答案</summary>

- `std::ifstream` / `ofstream` — 文件自动关闭
- `std::lock_guard` / `unique_lock` — 互斥锁自动解锁
- `std::vector` / `string` — 堆内存自动释放
- `std::thread` — join 或 detach 自动处理

</details>
