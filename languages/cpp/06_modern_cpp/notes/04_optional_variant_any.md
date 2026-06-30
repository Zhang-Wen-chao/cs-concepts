# std::optional、std::variant、std::any、std::expected

## 1. std::optional — 可能存在的值

代替指针传参/返回来"表示值可能没有"：

```cpp
#include <iostream>
#include <optional>

// 安全的除法：除零返回空
std::optional<double> safe_div(double a, double b) {
    if (b == 0.0) return std::nullopt;
    return a / b;
}

int main() {
    auto result = safe_div(10, 0);

    if (result.has_value()) {
        std::cout << *result << '\n';        // 解引用
        std::cout << result.value() << '\n'; // 抛异常 if empty
    } else {
        std::cout << "division by zero\n";
    }

    // C++17 的 value_or：提供默认值
    auto res2 = safe_div(10, 3);
    std::cout << res2.value_or(-1.0) << '\n';  // 3.333...
}
```

**何时用**：以往用 `T*`（nullptr 表示不存在）或 `bool` + 输出参数，现在用 `optional<T>`。

---

## 2. std::variant — 类型安全的联合体

代替 C 的 `union`，可以存放多种类型中的一种，且自动追踪当前类型：

```cpp
#include <iostream>
#include <variant>
#include <string>

int main() {
    std::variant<int, double, std::string> v;

    v = 42;
    std::cout << std::get<int>(v) << '\n';    // 42

    v = 3.14;
    std::cout << std::get<double>(v) << '\n'; // 3.14

    v = "hello"s;
    std::cout << std::get<std::string>(v) << '\n';  // hello

    // 安全访问：如果类型不对返回空 optional
    if (auto* p = std::get_if<int>(&v)) {
        std::cout << "int: " << *p << '\n';
    } else {
        std::cout << "not an int\n";
    }

    // 用 visit 统一处理
    auto visitor = [](auto&& arg) {
        std::cout << arg << '\n';
    };
    std::visit(visitor, v);  // 自动匹配当前类型
}
```

---

## 3. std::any — 可以装任意类型

```cpp
#include <iostream>
#include <any>

int main() {
    std::any a = 42;
    a = 3.14;
    a = std::string("hello");

    try {
        std::cout << std::any_cast<int>(a) << '\n';  // ❌ 抛出 bad_any_cast
    } catch (const std::bad_any_cast& e) {
        std::cout << "wrong type: " << e.what() << '\n';
    }

    // 安全做法
    if (auto* p = std::any_cast<std::string>(&a)) {
        std::cout << *p << '\n';
    }
}
```

**注意**：`std::any` 会动态分配内存（RTTI 实现），性能开销大。仅在运行时类型完全未知时使用。

---

## 4. std::expected — 返回结果或错误（C++23）

```cpp
#include <iostream>
#include <expected>
#include <string>

enum class Error { DivByZero, Overflow };

std::expected<double, Error> safe_div(double a, double b) {
    if (b == 0.0) return std::unexpected(Error::DivByZero);
    return a / b;
}

int main() {
    auto res = safe_div(10, 0);

    if (res) {
        std::cout << *res << '\n';
    } else {
        switch (res.error()) {
            case Error::DivByZero:
                std::cout << "divide by zero!\n";
                break;
            case Error::Overflow:
                std::cout << "overflow!\n";
                break;
        }
    }

    // 链式操作：and_then / transform
    auto chain = safe_div(10, 2)
        .and_then([](double v) { return safe_div(v, 5); })
        .or_else([](Error) -> std::expected<double, Error> {
            return 0.0;
        });
}
```

---

## 总结

| 类型 | 用途 | 引入 |
|------|------|------|
| `optional<T>` | 值可能存在也可能不存在 | C++17 |
| `variant<Ts...>` | 类型安全的联合体 | C++17 |
| `any` | 可存储任意类型 | C++17 |
| `expected<T, E>` | 返回结果或错误（类似 Rust 的 Result） | C++23 |
