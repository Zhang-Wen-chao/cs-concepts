# nullptr 替代 NULL、override/final 关键字

## 1. nullptr — C++11 的空指针字面量

在 C++03 里用 `NULL` 或 `0` 表示空指针，但这俩其实都是整数类型，可能引发歧义：

```cpp
#include <iostream>

void f(int)   { std::cout << "int\n"; }
void f(char*) { std::cout << "pointer\n"; }

int main() {
    f(0);       // 调用 f(int)
    f(NULL);    // 也是 f(int)！因为 NULL 是 0 的宏
    f(nullptr); // 调用 f(char*) — 正确！
}
```

`nullptr` 的类型是 `std::nullptr_t`，可以隐式转换为任何指针类型，但不能转为整数。

**最佳实践**：永远用 `nullptr`，不用 `NULL` 或 `0` 表示空指针。

---

## 2. override — 显式标记重写

C++11 引入 `override` 关键字，让编译器帮你检查是否真的在重写虚函数：

```cpp
class Base {
public:
    virtual void foo() {}
    virtual void bar(int) {}
};

class Derived : public Base {
public:
    void foo() override {}       // ✅ 正确重写
    void bar(double) override {} // ❌ 编译错误：Base 没有 bar(double)
    // 不加 override → 编译器不报错，但这是重载而非重写（隐藏了 Base::bar）
};
```

**加 `override` 的好处**：
- 防止因为参数类型写错而意外隐藏基类函数
- 给读者明确语义——"这一定是在重写虚函数"
- 编译器帮你兜底

---

## 3. final — 禁止继承或禁止重写

`final` 可以修饰类或虚函数：

```cpp
class Base {
public:
    virtual void step1() {}
    virtual void step2() {}
};

class Intermediate final : public Base {
public:
    void step1() override final {}  // ✓ 可重写，但禁止后续派生类再重写
    void step2() override {}
};

// class Derived : public Intermediate {};  ❌ 无法继承 Intermediate（final 类）
```

**用途**：
- `final` 类：设计继承树时封顶，防止意外扩展
- `final` 虚函数：模板方法模式中固定某些步骤

---

## 4. 组合使用

```cpp
class Interface {
public:
    virtual ~Interface() = default;
    virtual void doit() = 0;
};

class Implementation final : public Interface {
public:
    void doit() override {
        // ...
    }
};
```

---

## 总结

| 关键字 | 作用 | 误用后果 |
|--------|------|---------|
| `nullptr` | 类型安全的空指针 | 无（应一直使用） |
| `override` | 检查是否真的在重写 | 编译错误 |
| `final` | 禁止继承或重写 | 编译错误 |
