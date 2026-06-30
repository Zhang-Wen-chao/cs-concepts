# 虚函数表

## 1. 什么是 vtable？

每个有虚函数的类，编译器会生成一张 **虚函数表**（virtual table）。对象开头藏一个 `vptr` 指针指向这张表。

```cpp
class Base {
public:
    virtual void f1() { std::cout << "Base::f1\n"; }
    virtual void f2() { std::cout << "Base::f2\n"; }
    void f3() { std::cout << "Base::f3\n"; }        // 非虚，不入表
};

class Derived : public Base {
public:
    void f1() override { std::cout << "Derived::f1\n"; }
    virtual void f4() { std::cout << "Derived::f4\n"; }
};
```

```
Derived 对象内存布局（64位）：
┌─────────┐
│ vptr    │ ──→ vtable: [Derived::f1, Base::f2, Derived::f4]
│ Base 成员│
│ ...     │
└─────────┘
```

## 2. RTTI 与 type_info

RTTI（Run-Time Type Information）使得运行时可以问"你到底是什么类型"。

```cpp
#include <typeinfo>

void identify(const Animal& a) {
    // typeid 返回 const std::type_info&
    std::cout << typeid(a).name() << "\n";

    if (typeid(a) == typeid(Dog)) {
        std::cout << "It's a Dog!\n";
    }
}
```

RTTI 信息通常放在 vtable 的第一个槽位，所以开启 RTTI 有**内存开销**（每个 vtable 多一个指针）。部分场景用 `-fno-rtti` 关闭以减小体积。

## 3. dynamic_cast 的原理

```cpp
Base* bp = new Derived();
// dynamic_cast 在运行时查 vtable 确认类型兼容性
if (Derived* dp = dynamic_cast<Derived*>(bp)) {
    // 安全转换成功
} else {
    // 转换失败返回 nullptr
}
```

- **上行转换**（派生→基类）：静态就能确定，`dynamic_cast` 等同于 `static_cast`
- **下行转换**（基类→派生）：需要查 vtable 做运行时检查，有开销
- 转换失败：指针返回 `nullptr`，引用抛出 `std::bad_cast`

## 4. 性能开销

虚函数调用和普通函数调用比：

| 操作 | 开销 |
|------|------|
| 普通函数 | 一条 call 指令 |
| 虚函数 | 读 vptr → 查 vtable → 跳转（2次间接寻址） |
| dynamic_cast | 查 vtable 做类型遍历，可能涉及字符串比较 |
| typeid | 从 vtable 读类型信息 |

**量化**：虚函数 ≈ 普通函数慢 5-20%。但在绝大多数业务代码中，**这不是瓶颈**。内联失效才是更大的影响——虚函数不能内联。

## 关键点总结

- 每一个有虚函数的类 **一张 vtable**，每个对象 **一个 vptr**
- 虚函数调用 = 两级间接（vptr → vtable → 函数地址）
- RTTI（typeid/dynamic_cast）依赖 vtable 中的类型信息
- `dynamic_cast` 是**安全的**下行转换，但比 `static_cast` 慢
- 虚函数不能被 **内联优化**，热路径上要注意
