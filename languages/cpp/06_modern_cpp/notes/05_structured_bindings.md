# 结构化绑定、if/switch with init、折叠表达式

## 1. 结构化绑定（C++17）

把 pair、tuple、struct 的成员"拆开"赋值给多个变量：

```cpp
#include <iostream>
#include <tuple>
#include <map>
#include <set>

std::tuple<int, double, std::string> get_record() {
    return {42, 3.14, "hello"};
}

int main() {
    // 拆 tuple
    auto [id, score, name] = get_record();
    std::cout << id << ' ' << score << ' ' << name << '\n';

    // 拆 pair（关联容器 insert 返回值）
    std::map<int, std::string> m;
    auto [iter, inserted] = m.insert({1, "one"});
    if (inserted) {
        std::cout << "inserted: " << iter->second << '\n';
    }

    // 拆 struct（成员按声明顺序绑定）
    struct Point { double x, y, z; };
    Point p{1.0, 2.0, 3.0};
    auto [x, y, z] = p;

    // 数组也能拆
    int arr[] = {10, 20, 30};
    auto [a, b, c] = arr;
}
```

**本质**：编译器生成匿名变量，通过 `std::tuple_size` / `std::tuple_element` 拆包，不是运行时操作。

**注意**：不能嵌套 `auto& [a, [b, c]]`（C++ 不支持递归结构化绑定）。

---

## 2. if/switch with init（C++17）

条件判断的同时初始化一个变量，变量作用域限制在 if/switch 块内：

```cpp
#include <iostream>
#include <map>

int main() {
    std::map<int, std::string> m{{1, "one"}, {2, "two"}};

    // 查找 + 判断，变量 it 只在 if 范围内有效
    if (auto it = m.find(1); it != m.end()) {
        std::cout << "found: " << it->second << '\n';
    } else {
        std::cout << "not found\n";
    }  // it 在这里失效

    // switch 同理
    switch (char c = getchar(); c) {
        case 'y': std::cout << "yes\n"; break;
        case 'n': std::cout << "no\n"; break;
        default:  std::cout << "unknown\n"; break;
    }  // c 在这里失效
}
```

**好处**：缩小变量作用域，避免变量污染外层。

---

## 3. 折叠表达式（C++17）

变参模板中对参数包进行二元运算的简洁语法：

```cpp
#include <iostream>

// 一元右折叠 (args + ...)  展开为 (a + (b + (c + ...)))
template <typename... Args>
auto sum_right(Args... args) {
    return (args + ...);
}

// 一元左折叠 (... + args)  展开为 (((a + b) + c) + ...)
template <typename... Args>
auto sum_left(Args... args) {
    return (... + args);
}

// 带初始值的二元折叠
template <typename... Args>
auto sum_init(Args... args) {
    return (args + ... + 0);  // 空包时返回 0
}

// 实战：打印所有参数
template <typename... Args>
void print_all(Args... args) {
    (std::cout << ... << args) << '\n';  // 左折叠
}

// 逗号折叠：对每个元素做操作
template <typename... Args>
void for_each(Args... args) {
    ((std::cout << args << ' '), ...);  // 逗号运算符是二元！
}

int main() {
    std::cout << sum_right(1, 2, 3, 4) << '\n';  // (1 + (2 + (3 + 4))) = 10
    std::cout << sum_left(1, 2, 3, 4)  << '\n';  // (((1 + 2) + 3) + 4) = 10
    std::cout << sum_init() << '\n';             // 0（有初始值兜底）

    print_all(1, " hello ", 3.14);
    for_each(1, 2, 3);
}
```

---

## 总结

| 特性 | 引入 | 一句话 |
|------|------|--------|
| 结构化绑定 | C++17 | 把 tuple/pair/struct 拆成独立变量 |
| if/switch init | C++17 | 条件中定义变量，作用域受限 |
| 折叠表达式 | C++17 | `(arg + ...)` 简洁展开变参模板 |
