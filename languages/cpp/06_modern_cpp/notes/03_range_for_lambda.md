# 范围 for 循环、lambda 表达式

> For a lambda-only deep dive with more examples, see `_supplementary/05_lambda.md`

## 1. 范围 for 循环

C++11 引入类似 Python 的 `for x in list` 语法：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 按值遍历（拷贝）
    for (int x : v) {
        std::cout << x << ' ';
    }

    // 按引用修改
    for (auto& x : v) {
        x *= 2;
    }

    // const 引用（大对象避免拷贝）
    for (const auto& x : v) {
        std::cout << x << ' ';
    }

    // 结构化绑定 + 范围 for（C++17）
    std::map<int, std::string> m{{1, "a"}, {2, "b"}};
    for (const auto& [k, v] : m) {
        std::cout << k << " -> " << v << '\n';
    }
}
```

**原理**：编译器展开为 `begin()`/`end()` 迭代器循环，任何有 `begin`/`end` 的类型都能用。

---

## 2. Lambda 表达式

Lambda 是**匿名可调用对象**，C++11 引入：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9};

    // 基本语法：[捕获](参数) -> 返回类型 { 函数体 }
    auto print = [](int x) { std::cout << x << ' '; };

    // 排序：降序
    std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

    // 捕获外部变量
    int threshold = 3;
    auto count = std::count_if(v.begin(), v.end(), 
        [threshold](int x) { return x > threshold; });

    // 引用捕获
    int sum = 0;
    std::for_each(v.begin(), v.end(), [&sum](int x) { sum += x; });
    std::cout << "sum = " << sum << '\n';
}
```

---

## 3. 捕获模式详解

```cpp
int a = 1, b = 2;

auto f1 = [=]() { return a + b; };      // 全部按值捕获
auto f2 = [&]() { a++; b++; };          // 全部按引用捕获
auto f3 = [=, &b]() { b++; return a; }; // 混合：b 引用，a 值
auto f4 = [&, a]() { return a + b; };   // 混合：a 值，其他引用
```

---

## 4. mutable 和泛型 Lambda

```cpp
#include <iostream>

int main() {
    int count = 0;

    // mutable：允许修改按值捕获的副本（不影响外部）
    auto counter = [count]() mutable { 
        return ++count; 
    };
    std::cout << counter() << '\n';  // 1
    std::cout << counter() << '\n';  // 2
    std::cout << count << '\n';      // 0（外部未变）

    // 泛型 Lambda（C++14）
    auto add = [](auto a, auto b) { return a + b; };
    std::cout << add(1, 2)   << '\n';  // int: 3
    std::cout << add(1.5, 2) << '\n';  // double: 3.5
}
```

**原理**：Lambda 本质是编译器生成的匿名类，重载了 `operator()`。

---

## 总结

| 特性 | 引入版本 | 关键点 |
|------|---------|--------|
| 范围 for | C++11 | 需要 begin/end |
| Lambda 基础 | C++11 | `[捕获](参数){体}` |
| 泛型 Lambda | C++14 | `auto` 参数 |
| 结构化绑定 + for | C++17 | `[&, var]` 混合捕获 |
