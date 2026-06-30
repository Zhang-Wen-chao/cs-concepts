# auto 类型推导、decltype、尾置返回类型

## 1. auto — 编译器自动推导类型

C++11 引入 `auto`，让编译器根据初始化表达式自动推导变量类型。

```cpp
#include <iostream>
#include <vector>
#include <map>

int main() {
    auto i = 42;           // int
    auto d = 3.14;         // double
    auto s = "hello";      // const char*
    auto v = std::vector{1, 2, 3};  // std::vector<int>

    // 迭代器再也不用写长长的一串
    std::map<int, std::string> m{{1, "one"}, {2, "two"}};
    for (auto it = m.begin(); it != m.end(); ++it) {
        std::cout << it->first << ": " << it->second << '\n';
    }
}
```

**注意**：`auto` 会忽略引用和顶层 const（除非显式写 `const auto&`）。

```cpp
const int ci = 0;
auto a1 = ci;   // int（const 被丢弃）
auto& a2 = ci;  // const int&（引用保留 const）
```

---

## 2. decltype — 获取表达式的类型

`decltype` 在编译期返回表达式的精确类型，**不会丢弃引用或 const**。

```cpp
int x = 0;
const int& rx = x;

decltype(x)  a = x;   // int
decltype(rx) b = x;   // const int&
decltype((x)) c = x;  // int&（双层括号变成左值引用！）
```

**关键区别**：`decltype((x))` 因为加了括号，`x` 被视为左值表达式，推导为 `int&`。这是 C++ 的一个坑。

---

## 3. 尾置返回类型（Trailing Return Type）

当返回类型依赖于参数类型时，用 `->` 把返回类型写在后面：

```cpp
// 模板中非常有用
template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// C++14 起可以省略尾置，直接用 auto
template <typename T, typename U>
auto add14(T a, U b) {
    return a + b;
}
```

---

## 4. C++14 的 decltype(auto)

C++14 新增 `decltype(auto)`，让 `auto` 按 `decltype` 的规则推导（保留引用和 const）：

```cpp
int x = 42;
int& get_ref() { return x; }

decltype(auto) r1 = get_ref();  // int&
auto           r2 = get_ref();  // int（引用被丢弃！）
```

---

## 总结

| 关键字 | 推导规则 | 典型用途 |
|--------|---------|---------|
| `auto` | 忽略引用/顶层 const | 变量声明、迭代器 |
| `decltype` | 保留引用/const | 模板返回类型 |
| `decltype(auto)` | auto 用 decltype 规则 | 完美转发返回值 |
| 尾置返回类型 | `-> type` 语法 | 依赖参数类型的返回值 |
