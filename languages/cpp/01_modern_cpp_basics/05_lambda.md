# Lambda 表达式详解

> 函数式编程，让代码更简洁

## 🎯 本课目标

- 理解 Lambda 表达式的语法
- 掌握捕获列表的用法
- 学会在算法中使用 Lambda
- 理解 Lambda 的底层原理
- 避免常见的 Lambda 陷阱

---

## 1️⃣ 什么是 Lambda？

### 问题：需要简单的函数

```cpp
// 旧方式：定义命名函数
bool is_even(int x) {
    return x % 2 == 0;
}

std::vector<int> vec = {1, 2, 3, 4, 5};
auto it = std::find_if(vec.begin(), vec.end(), is_even);
```

**问题：**
- `is_even` 只用一次，却要单独定义
- 代码分散，不直观

### Lambda：匿名函数

```cpp
// 新方式：Lambda 表达式
std::vector<int> vec = {1, 2, 3, 4, 5};
auto it = std::find_if(vec.begin(), vec.end(),
                       [](int x) { return x % 2 == 0; });
//                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                              Lambda 表达式
```

**Lambda = 匿名函数 = 就地定义的小函数**

---

## 2️⃣ Lambda 语法

### 完整语法

```cpp
[捕获列表](参数列表) mutable noexcept -> 返回类型 { 函数体 }
```

### 最简单的 Lambda

```cpp
// 无参数，无返回值
auto f = []() { std::cout << "Hello" << std::endl; };
f();  // 调用

// 简化：省略空参数列表
auto f2 = [] { std::cout << "Hello" << std::endl; };
f2();
```

### 带参数的 Lambda

```cpp
// 有参数，返回类型自动推导
auto add = [](int a, int b) { return a + b; };
std::cout << add(3, 4) << std::endl;  // 输出：7

// 显式指定返回类型
auto divide = [](int a, int b) -> double {
    return static_cast<double>(a) / b;
};
```

### 语法详解

```cpp
[capture](params) -> return_type { body }
 ↑       ↑        ↑               ↑
 捕获    参数     返回类型        函数体

// 示例
auto lambda = [x](int y) -> int { return x + y; };
//            ↑   ↑      ↑       ↑
//            捕获 参数   返回类型 函数体
```

---

## 3️⃣ 捕获列表（Capture）

### 捕获外部变量

```cpp
int x = 10;

// []：不捕获任何变量
auto f1 = []() {
    // std::cout << x;  // ❌ 错误：x 不可见
};

// [x]：按值捕获 x
auto f2 = [x]() {
    std::cout << x << std::endl;  // ✅ 可以访问 x
};

// [&x]：按引用捕获 x
auto f3 = [&x]() {
    x = 20;  // ✅ 可以修改 x
};
```

### 捕获方式总结

```cpp
int a = 1, b = 2;

// [a]：按值捕获 a
auto f1 = [a]() { std::cout << a; };

// [&a]：按引用捕获 a
auto f2 = [&a]() { a = 10; };

// [a, &b]：a 按值，b 按引用
auto f3 = [a, &b]() { b = a + 10; };

// [=]：按值捕获所有外部变量
auto f4 = [=]() { std::cout << a + b; };

// [&]：按引用捕获所有外部变量
auto f5 = [&]() { a = 10; b = 20; };

// [=, &b]：默认按值，b 按引用
auto f6 = [=, &b]() { b = a + 10; };

// [&, a]：默认按引用，a 按值
auto f7 = [&, a]() { b = a + 10; };
```

### 按值 vs 按引用

```cpp
int x = 10;

// 按值捕获：拷贝 x
auto f1 = [x]() {
    std::cout << x << std::endl;  // 输出：10
};
x = 20;
f1();  // 输出：10（捕获时的值）

// 按引用捕获：引用 x
auto f2 = [&x]() {
    std::cout << x << std::endl;
};
x = 30;
f2();  // 输出：30（当前的值）
```

### mutable 关键字

```cpp
int x = 10;

// 按值捕获默认是 const
auto f1 = [x]() {
    // x = 20;  // ❌ 错误：不能修改
};

// mutable：可以修改捕获的值（但不影响原变量）
auto f2 = [x]() mutable {
    x = 20;  // ✅ 可以修改（Lambda 内部的拷贝）
    std::cout << "Lambda 内: " << x << std::endl;
};

f2();  // 输出：Lambda 内: 20
std::cout << "外部: " << x << std::endl;  // 输出：外部: 10
```

---

## 4️⃣ Lambda 与标准算法

### std::for_each

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};

// 打印每个元素
std::for_each(vec.begin(), vec.end(),
              [](int x) { std::cout << x << " "; });
```

### std::find_if

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};

// 查找第一个偶数
auto it = std::find_if(vec.begin(), vec.end(),
                       [](int x) { return x % 2 == 0; });

if (it != vec.end()) {
    std::cout << "找到: " << *it << std::endl;  // 输出：2
}
```

### std::sort

```cpp
std::vector<int> vec = {5, 2, 8, 1, 9};

// 降序排序
std::sort(vec.begin(), vec.end(),
          [](int a, int b) { return a > b; });
// 结果：9 8 5 2 1
```

### std::count_if

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 统计大于 5 的数
int count = std::count_if(vec.begin(), vec.end(),
                          [](int x) { return x > 5; });
// count = 5
```

### std::transform

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};
std::vector<int> result(vec.size());

// 每个元素乘以 2
std::transform(vec.begin(), vec.end(), result.begin(),
               [](int x) { return x * 2; });
// result = {2, 4, 6, 8, 10}
```

### std::remove_if

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5, 6};

// 删除所有偶数
vec.erase(
    std::remove_if(vec.begin(), vec.end(),
                   [](int x) { return x % 2 == 0; }),
    vec.end()
);
// vec = {1, 3, 5}
```

---

## 5️⃣ Lambda 的类型和存储

### Lambda 的类型

```cpp
// Lambda 有唯一的类型（编译器生成）
auto f1 = [](int x) { return x + 1; };
auto f2 = [](int x) { return x + 1; };

// f1 和 f2 类型不同！
// decltype(f1) != decltype(f2)
```

### 用 std::function 存储

```cpp
#include <functional>

// std::function 可以存储任何可调用对象
std::function<int(int)> f1 = [](int x) { return x + 1; };
std::function<int(int)> f2 = [](int x) { return x + 2; };

// f1 和 f2 类型相同
std::vector<std::function<int(int)>> funcs = {f1, f2};
```

### auto vs std::function

```cpp
// ✅ 推荐：auto（零开销）
auto f1 = [](int x) { return x + 1; };

// ⚠️ 有开销：std::function
std::function<int(int)> f2 = [](int x) { return x + 1; };

// 原因：std::function 有类型擦除的开销
```

---

## 6️⃣ Lambda 的底层原理

### Lambda 是什么？

```cpp
// Lambda
auto lambda = [x](int y) { return x + y; };

// 编译器生成的类（简化版）
class __lambda_123 {
    int x_;  // 捕获的变量

public:
    __lambda_123(int x) : x_(x) {}  // 构造函数

    int operator()(int y) const {   // 重载 operator()
        return x_ + y;
    }
};

auto lambda = __lambda_123(x);
```

**Lambda = 编译器生成的仿函数（Functor）**

### 捕获的实现

```cpp
int a = 1, b = 2;

// [a, &b]：按值捕获 a，按引用捕获 b
auto lambda = [a, &b]() { return a + b; };

// 编译器生成：
class __lambda {
    int a_;     // 按值：存储拷贝
    int& b_;    // 按引用：存储引用

public:
    __lambda(int a, int& b) : a_(a), b_(b) {}

    int operator()() const {
        return a_ + b_;
    }
};
```

---

## 7️⃣ 泛型 Lambda（C++14）

### 参数类型用 auto

```cpp
// C++11：必须指定类型
auto f1 = [](int x) { return x + 1; };

// C++14：可以用 auto
auto f2 = [](auto x) { return x + 1; };

f2(10);      // int
f2(3.14);    // double
f2("hello"); // 编译错误：const char* 不支持 +
```

### 泛型 Lambda 的原理

```cpp
auto lambda = [](auto x, auto y) { return x + y; };

// 编译器生成：
class __lambda {
public:
    template<typename T, typename U>
    auto operator()(T x, U y) const {
        return x + y;
    }
};
```

---

## 8️⃣ 常见陷阱

### 陷阱 1：悬空引用

```cpp
std::function<int()> create_lambda() {
    int x = 10;
    return [&x]() { return x; };  // ❌ 危险：x 的生命周期结束
}

auto f = create_lambda();
int result = f();  // 💥 未定义行为：x 已经销毁
```

**修复：按值捕获**

```cpp
std::function<int()> create_lambda() {
    int x = 10;
    return [x]() { return x; };  // ✅ 安全：拷贝 x
}
```

### 陷阱 2：[=] 捕获 this

```cpp
class Widget {
    int value_ = 42;

public:
    auto create_lambda() {
        // [=]：捕获 this 指针（不是 value_）
        return [=]() { return value_; };
    }
};

// 危险：如果 Widget 对象销毁，Lambda 访问 value_ 会出错
```

**修复：显式捕获**

```cpp
auto create_lambda() {
    // C++14：显式按值捕获成员
    return [value = value_]() { return value; };

    // 或者按值捕获 this（C++17）
    return [*this]() { return value_; };
}
```

### 陷阱 3：按值捕获大对象

```cpp
std::vector<int> large_vec(1000000);

// ❌ 低效：拷贝整个 vector
auto f1 = [large_vec]() {
    return large_vec.size();
};

// ✅ 高效：引用
auto f2 = [&large_vec]() {
    return large_vec.size();
};

// ✅ 更好：只捕获需要的
auto f3 = [size = large_vec.size()]() {
    return size;
};
```

### 陷阱 4：mutable 不改变原变量

```cpp
int x = 10;

auto f = [x]() mutable {
    x = 20;  // 修改的是 Lambda 内部的拷贝
};

f();
std::cout << x << std::endl;  // 输出：10（不是 20）
```

---

## 9️⃣ 高级用法

### 初始化捕获（C++14）

```cpp
// 移动捕获
auto ptr = std::make_unique<int>(42);
auto f = [p = std::move(ptr)]() {
    return *p;
};

// ptr 现在是空的，所有权转移给 Lambda
```

### 立即调用的 Lambda（IIFE）

```cpp
// 复杂的初始化
int x = []() {
    if (some_condition) {
        return 42;
    } else {
        return 100;
    }
}();  // 立即调用
```

### Lambda 递归

```cpp
// C++14：用 std::function
std::function<int(int)> fib = [&fib](int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);
};

std::cout << fib(10) << std::endl;  // 55
```

---

## 🔟 最佳实践

### 1. 默认用 auto

```cpp
// ✅ 推荐
auto f = [](int x) { return x + 1; };

// ⚠️ 非必要不用 std::function
std::function<int(int)> f = [](int x) { return x + 1; };
```

### 2. 小心按引用捕获

```cpp
// ✅ Lambda 立即使用：可以按引用
std::vector<int> vec = {1, 2, 3};
std::for_each(vec.begin(), vec.end(),
              [&](int x) { std::cout << x; });

// ❌ Lambda 延迟使用：不要按引用
auto f = [&vec]() { return vec.size(); };  // 危险
// 如果 vec 销毁，f 会出错
```

### 3. 泛型 Lambda 替代模板函数

```cpp
// ❌ 旧方式：模板函数
template<typename T>
void print(const T& x) {
    std::cout << x << std::endl;
}

// ✅ 新方式：泛型 Lambda
auto print = [](const auto& x) {
    std::cout << x << std::endl;
};
```

### 4. 初始化捕获替代按值捕获

```cpp
std::string s = "long string";

// ❌ 拷贝
auto f1 = [s]() { return s.size(); };

// ✅ 移动
auto f2 = [s = std::move(s)]() { return s.size(); };
```

---

## 🎯 总结

### Lambda 语法

```cpp
[capture](params) -> return_type { body }

// 示例
auto f = [x, &y](int a, int b) -> int {
    return x + y + a + b;
};
```

### 捕获方式

```cpp
[]          不捕获
[x]         按值捕获 x
[&x]        按引用捕获 x
[=]         按值捕获所有
[&]         按引用捕获所有
[x, &y]     x 按值，y 按引用
[=, &y]     默认按值，y 按引用
```

### 使用场景

```cpp
// 1️⃣ 配合算法
std::sort(vec.begin(), vec.end(),
          [](int a, int b) { return a > b; });

// 2️⃣ 回调函数
button.onClick([this]() {
    this->handleClick();
});

// 3️⃣ 短小的函数
auto is_positive = [](int x) { return x > 0; };
```

### 核心原则

1. **Lambda = 就地定义的小函数**
2. **底层 = 编译器生成的仿函数**
3. **按值捕获 = 拷贝，按引用捕获 = 引用**
4. **小心悬空引用（按引用捕获已销毁的对象）**
5. **默认用 auto，非必要不用 std::function**

---

## 🚀 下一步

学完 Lambda 后，接下来学习：
1. **模板基础**（泛型编程）
2. **并发编程**（多线程）
3. **函数式编程**（深入）

**配套实践代码：** [practices/05_lambda.cpp](practices/05_lambda.cpp)
