# 1.10 const / 引用 / move — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

`const int& x = 42;` 能编译吗？为什么？

<details>
<summary>答案</summary>

能。`const T&` 可以绑定到**右值（临时对象）**。编译器会为 `42` 创建一个临时 int，然后 `x` 引用它。这叫 **const 引用延长生命周期**。

```cpp
const int& x = 42;        // ✅ 存活
int& y = 42;              // ❌ 不能编译 — 非常量引用不能绑右值
```

这是一个非常常用的写法，也是为什么函数参数推荐 `const T&` 的原因之一 — 你可以传临时对象给它。

</details>

## Q2

`std::move` 做了什么？它真的移动了数据吗？

<details>
<summary>答案</summary>

**`std::move` 没有移动任何数据。** 它只是把参数**转型为右值引用**（`T&&`），让编译器知道"这东西我不用了，可以偷走它的资源"。

```cpp
std::string a = "hello";
std::string b = std::move(a);  // move 只是 cast，实际移动是 string 的移动构造函数干的
```

移动后 `a` 处于"有效但未指定"状态（通常是空字符串），不再用 `a` 除非重新赋值。

</details>

## Q3

移动语义在 Python 里有对应吗？为什么 C++ 需要它？

<details>
<summary>答案</summary>

Python 没有移动语义。Python 所有对象都在堆上，变量名只是引用，赋值 = 引用计数 +1，不需要"移动"。

C++ 需要移动语义因为**值语义 + 大规模数据**：

```cpp
std::vector<int> createBigVector() {
    std::vector<int> v(1000000);
    return v;  // C++11 前会拷贝整个 vector，现在会移动（或 NRVO）
}
```

没有移动语义 → 每次函数返回大对象都得拷贝全部数据。移动语义 → 偷指针，近乎零开销。

</details>

## Q4

右值引用（`T&&`）和万能引用（`T&&` 在模板里）怎么区分？

<details>
<summary>答案</summary>

看上下文：

```cpp
void foo(int&& x);                // 右值引用 — 只能绑右值
template <typename T>
void bar(T&& x);                  // 万能引用 — 左右都能绑

auto&& x = expr;                  // 万能引用
```

**万能引用**只有在类型推导的上下文中才出现（模板、`auto&&`、`decltype(auto)`）。具体是什么取决于传入的参数：
- 传左值 → `T` 推导为 `T&` → 变为左值引用
- 传右值 → `T` 推导为 `T` → 变为右值引用

这叫"引用折叠"（reference collapsing）。

</details>

## Q5

什么时候该用 `const&`，什么时候该传值（`T`）？

<details>
<summary>答案</summary>

```cpp
// 只读查看 — const&
void print(const std::string& s);

// 要拷贝一份存储 — 传值然后 move
class Widget {
    std::string name_;
public:
    Widget(std::string name) : name_(std::move(name)) {}  // 传值 + move
};
```

**黄金准则**：
- 只读 → `const T&`
- 需要存储一份 → 传 `T` 值，然后用 `std::move` 构造到成员
- 小类型（int, char, bool）→ 直接传值，不占引用

</details>

## Q6

`const` 在 C++ 里有哪些用法？

<details>
<summary>答案</summary>

| 写法 | 含义 |
|---|---|
| `const int x = 5;` | x 不能修改 |
| `const int* p;` | 指向的内容不能改（指向可以换） |
| `int* const p;` | 指针本身不能改（指向的内容可以改） |
| `const int* const p;` | 指针和内容都不能改 |
| `const string& s;` | 引用，不能通过 s 改内容 |
| `void foo() const;` | 成员函数不修改对象状态 |
| `const T& operator[](size_t i) const;` | 重载常量和非常量版本 |

**C++ const 的王道**: 能 const 就 const。编译器帮你检查，跑起来零开销。

</details>
