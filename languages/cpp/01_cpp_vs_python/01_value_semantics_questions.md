# 1.4 值语义 — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

值语义（value semantics）和引用语义（reference semantics）是什么？

<details>
<summary>答案</summary>

- **值语义**：赋值、传参时**拷贝整个数据**。两份各自独立，改一个不影响另一个。
  - C++ 默认行为：`int a = b`，`vector v = w` 都是拷贝
- **引用语义**：赋值、传参时只传**引用/指针**。两份指向同一块数据，改一个另一个也变。
  - Python 默认行为：`a = b` 就是给同一对象贴了个新标签

```python
# Python = 引用语义
a = [1, 2, 3]
b = a
b.append(4)   # a 也是 [1, 2, 3, 4] — 同一份数据
```

```cpp
// C++ = 值语义
std::vector<int> a = {1, 2, 3};
auto b = a;       // 拷贝
b.push_back(4);   // a 还是 [1, 2, 3] — 两份独立
```

</details>

## Q2

对象切片（object slicing）是什么？

<details>
<summary>答案</summary>

用值语义传子类对象时，编译器只拷贝了基类部分，子类的东西丢了。

```cpp
struct Base { int x; };
struct Derived : Base { int y; };

void foo(Base b) {}   // 值传参

Derived d;
foo(d);   // 传入的 b 只有 x，y 被切掉了
```

**解决**：传指针或引用 `void foo(const Base& b)`。

</details>

## Q3

Python 里 `def foo(lst): lst.append(4)` 会改到调用方的列表。C++ 想要同样行为怎么写？

<details>
<summary>答案</summary>

```cpp
// ❌ 值语义 — 传进来的是拷贝，改不到外面的
void foo(std::vector<int> lst) {
    lst.push_back(4);  // 只改了局部拷贝
}

// ✅ 传引用
void foo(std::vector<int>& lst) {
    lst.push_back(4);  // 改到了原来的
}
```

`&` 就是 C++ 的"引用语义开关"。默认值语义，你想传引用必须显式写 `&`。

</details>

## Q4

C++ 函数参数的最佳实践是什么？（`T`, `const T&`, `T&` 什么时候用哪个？）

<details>
<summary>答案</summary>

| 方式 | 传参行为 | 改外面？拷贝？ | 什么时候用 |
|---|---|---|---|
| `T` | 拷贝 | 不改，花销大 | 小类型（int, char）或确实要拷贝时 |
| `const T&` | 引用 | 不改，不拷贝 | **大部分情况首选** — 只读不写 |
| `T&` | 引用 | 能改，不拷贝 | 需要修改调用方的数据 |

**黄金规则**：参数只读就用 `const T&`。这是 C++ 最常用的模式，Python 的 `def foo(lst)` 在 C++ 写成 `void foo(const std::vector<int>& lst)`。

</details>

## Q5

C++ 什么时候用指针，什么时候用引用？

<details>
<summary>答案</summary>

| 场景 | 用指针 | 用引用 |
|---|---|---|
| 可能为空 | ✅ `T* p = nullptr` | ❌ 引用不能为空 |
| 需要重新绑定 | ✅ `p = &other` | ❌ 引用一生绑定一个 |
| 函数参数只读 | ❌ | ✅ `const T&` |
| 想改调用方 | ❌ | ✅ `T&`（指针也行但语法丑） |

**一句话**：能不用指针就不用指针。引用更安全（不可能是 null）。

</details>
