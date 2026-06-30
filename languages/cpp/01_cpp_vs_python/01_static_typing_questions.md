# 1.1 静态类型 — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

Python: `x = 42` 之后 `x = "hello"` 为什么可以？
C++: `int x = 42; x = "hello";` 为什么不行？

<details>
<summary>答案</summary>

Python 是**动态类型**：变量名只是标签，可以贴到任何类型的东西上。
C++ 是**静态类型**：变量声明时就定死了类型，后面不能换。`x` 声明成 `int`，就一直都是 `int`。

</details>

## Q2

C++ 的 `auto` 关键字是让类型变成动态吗？为什么？

<details>
<summary>答案</summary>

不是。`auto` 是**编译期推导**，编译器根据初始值推断类型，推导完就定死了，后面不能换类型。只是帮你省掉手写类型的功夫。

```cpp
auto x = 42;    // x 是 int
x = "hello";    // ❌ 编译错误，int 不能赋 string
```

</details>

## Q3

函数模板和 Python 的 duck typing 有什么区别？

<details>
<summary>答案</summary>

```cpp
template <typename T>
T add(T a, T b) { return a + b; }
```

- **模板**：编译期生成一份 int 版、一份 double 版、一份 string 版……类型不符直接编译失败
- **Python duck typing**：运行时才检测，传了什么就用什么的方法，没有的类型方法就 AttributeError

模板 = **编译期静态多态**，Python duck typing = **运行时动态派发**。

</details>

## Q4

以下代码能编译吗？为什么？

```cpp
auto x = 10;
x = 3.14;
```

<details>
<summary>答案</summary>

不能。`auto x = 10` 推导出 `x` 是 `int`，`3.14` 是 `double`。尝试把 `double` 赋给 `int`，可能有精度损失，**编译不通过**（少数编译器会给 warning + 窄化转换，但标准说应该报错或 warning）。

改成 `auto x = 3.14;`（推导为 double）或 `x = static_cast<int>(3.14);` 才行。

</details>

## Q5

`explicit` 关键字在构造函数前面有什么用？

<details>
<summary>答案</summary>

阻止编译器**隐式转换**。

```cpp
class String {
public:
    explicit String(int size);  // 不用 explicit 的话…
};

String s = 42;  // ❌ explicit 阻止了 "int → String" 的隐式构造
String s(42);   // ✅ 必须显式构造
```

不加 `explicit` 时 `String s = 42` 会隐式调用 `String(42)`，通常不是你想要的行为。

</details>

## Q6

`static_cast`、`dynamic_cast`、`reinterpret_cast`、`const_cast` 各在什么场景用？

<details>
<summary>答案</summary>

| cast | 场景 | 举例 |
|---|---|---|
| `static_cast` | 编译期已知安全的类型转换 | `int → double`，`void* → T*`，父子类上行 |
| `dynamic_cast` | 运行时安全检查的多态下行转换 | `Base* → Derived*`，失败返回 null |
| `reinterpret_cast` | 重新解释比特位（危险） | `int* → char*`，指针转整数 |
| `const_cast` | 去掉 const 属性（尽量别用） | `const int* → int*` |

**C 风格 `(int)3.14`** 在 C++ 里最好不用，它可能干出 `reinterpret_cast` 的事。

</details>
