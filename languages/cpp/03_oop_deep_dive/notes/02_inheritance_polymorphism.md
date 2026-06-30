# 继承与多态

## 1. 继承方式

```cpp
class Base {
public:    int pub = 1;
protected: int pro = 2;
private:   int pri = 3;
};

class PubChild  : public  Base {};  // pub→pub, pro→pro, pri 不可见
class ProChild  : protected Base {}; // pub→pro, pro→pro, pri 不可见
class PriChild  : private   Base {}; // pub→pri, pro→pri, pri 不可见
```

公有继承 ≈ "is-a" 关系。私有继承 ≈ "implemented-in-terms-of"（组合的替代）。

## 2. 虚函数与多态

```cpp
class Animal {
public:
    virtual void speak() const {          // virtual 让派生类可以覆盖
        std::cout << "?\n";
    }
    virtual ~Animal() = default;          // 基类析构必须虚，否则派生类对象析构不完整
};

class Dog : public Animal {
public:
    void speak() const override {         // override 显式声明覆盖，编译器帮忙检查
        std::cout << "Woof!\n";
    }
};

void letSpeak(const Animal& a) { a.speak(); }

int main() {
    Dog d;
    letSpeak(d);                          // 输出 Woof!  — 多态
    Animal a;
    letSpeak(a);                          // 输出 ?
}
```

**原理**：通过 `virtual` 函数表（vtable）在运行时动态分派。没有 `virtual` 的话，上面两行都输出 "?"。

## 3. override 和 final

```cpp
class Base {
    virtual void foo();
    virtual void bar();
};

class Derived : public Base {
    void foo() override;    // ✅ 正确，覆盖 Base::foo
    void bar() const override; // ❌ 编译错误：签名不匹配
};

class Sealed final {        // final 类：不能被继承
    // ...
};
```

- `override`：告诉编译器"我就是要覆盖"，写错了立即报错
- `final`：阻止派生类继续覆盖，或者阻止类被继承

## 4. 纯虚函数与抽象类

```cpp
class Shape {                               // 抽象类：不能实例化
public:
    virtual double area() const = 0;        // 纯虚函数
    virtual ~Shape() = default;
};

class Circle : public Shape {
    double r_;
public:
    Circle(double r) : r_(r) {}
    double area() const override { return 3.14159 * r_ * r_; }
};
```

- `= 0` 就是纯虚函数，表示"没有实现，派生类必须实现"
- 有纯虚函数的类 = 抽象类，不能创建对象
- 抽象类可以定义普通成员函数

## 关键点总结

- **继承方式**控制基类成员在派生类中的可见性
- **虚函数**实现运行时多态，需要 `virtual` + 指针/引用
- 基类 **析构函数** 必须 `virtual`，否则派生类资源泄漏
- `override` 是 C++11 的**安全习惯**，必须用
- 纯虚函数定义 **接口契约**，对应 Java interface / Go interface 的概念
