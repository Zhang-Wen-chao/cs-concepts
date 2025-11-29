# 模板基础

> 泛型编程，一次编写，处处复用

## 🎯 本课目标

- 理解模板的作用和原理
- 掌握函数模板和类模板
- 理解模板实例化
- 学会使用模板特化
- 避免常见的模板错误

---

## 1️⃣ 为什么需要模板？

### 问题：重复代码

```cpp
// 交换 int
void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// 交换 double（重复代码）
void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}

// 交换 string（重复代码）
void swap(std::string& a, std::string& b) {
    std::string temp = a;
    a = b;
    b = temp;
}

// 为每种类型都写一遍？太繁琐！
```

### 模板：一次编写，处处复用

```cpp
// 函数模板：适用于所有类型
template<typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// 使用
int x = 1, y = 2;
swap(x, y);  // 自动推导 T = int

double a = 1.5, b = 2.5;
swap(a, b);  // 自动推导 T = double

std::string s1 = "hello", s2 = "world";
swap(s1, s2);  // 自动推导 T = std::string
```

**模板 = 让编译器为每种类型生成代码**

---

## 2️⃣ 函数模板

### 基本语法

```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// 使用
int x = max(3, 5);           // T = int
double y = max(3.14, 2.71);  // T = double
```

### 完整语法

```cpp
template<typename T>  // 或 template<class T>
返回类型 函数名(参数列表) {
    函数体
}
```

**typename vs class：**
```cpp
template<typename T>  // ✅ 推荐（更清晰）
template<class T>     // ✅ 一样的（历史遗留）

// 两者完全等价
```

### 多个模板参数

```cpp
template<typename T1, typename T2>
void print_pair(const T1& a, const T2& b) {
    std::cout << a << ", " << b << std::endl;
}

// 使用
print_pair(42, "hello");      // T1 = int, T2 = const char*
print_pair(3.14, std::string("world"));  // T1 = double, T2 = string
```

### 显式指定类型

```cpp
template<typename T>
T add(T a, T b) {
    return a + b;
}

// 自动推导
auto x = add(1, 2);  // T = int

// 显式指定
auto y = add<double>(1, 2);  // T = double，结果是 3.0
```

### 返回类型推导

```cpp
// C++11：需要尾置返回类型
template<typename T1, typename T2>
auto add(T1 a, T2 b) -> decltype(a + b) {
    return a + b;
}

// C++14：自动推导
template<typename T1, typename T2>
auto add(T1 a, T2 b) {
    return a + b;  // 编译器自动推导返回类型
}

// 使用
auto result = add(1, 2.5);  // T1 = int, T2 = double, 返回 double
```

---

## 3️⃣ 类模板

### 基本语法

```cpp
template<typename T>
class Stack {
    std::vector<T> elements_;

public:
    void push(const T& elem) {
        elements_.push_back(elem);
    }

    void pop() {
        if (!elements_.empty()) {
            elements_.pop_back();
        }
    }

    T top() const {
        return elements_.back();
    }

    bool empty() const {
        return elements_.empty();
    }
};

// 使用：必须显式指定类型
Stack<int> int_stack;
int_stack.push(1);
int_stack.push(2);

Stack<std::string> string_stack;
string_stack.push("hello");
```

**注意：类模板不能自动推导类型（C++17 之前）**

```cpp
// ❌ C++14 及之前：编译错误
Stack s;  // 错误：缺少模板参数

// ✅ 必须显式指定
Stack<int> s;

// ✅ C++17：可以推导
Stack s{1, 2, 3};  // 推导为 Stack<int>
```

### 类模板的成员函数定义

```cpp
template<typename T>
class MyVector {
    T* data_;
    size_t size_;

public:
    MyVector(size_t size);
    void push_back(const T& value);
    T& operator[](size_t index);
};

// 类外定义成员函数
template<typename T>
MyVector<T>::MyVector(size_t size) : size_(size) {
    data_ = new T[size];
}

template<typename T>
void MyVector<T>::push_back(const T& value) {
    // ...
}

template<typename T>
T& MyVector<T>::operator[](size_t index) {
    return data_[index];
}
```

### 多个模板参数

```cpp
template<typename K, typename V>
class Map {
    std::vector<std::pair<K, V>> pairs_;

public:
    void insert(const K& key, const V& value) {
        pairs_.push_back({key, value});
    }

    V* find(const K& key) {
        for (auto& p : pairs_) {
            if (p.first == key) {
                return &p.second;
            }
        }
        return nullptr;
    }
};

// 使用
Map<std::string, int> age_map;
age_map.insert("Alice", 25);
age_map.insert("Bob", 30);
```

---

## 4️⃣ 模板实例化

### 什么是实例化？

```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

int x = max(3, 5);        // 实例化：max<int>
double y = max(3.14, 2.71);  // 实例化：max<double>

// 编译器生成：
int max(int a, int b) { return (a > b) ? a : b; }
double max(double a, double b) { return (a > b) ? a : b; }
```

**实例化 = 编译器用具体类型替换模板参数，生成真正的代码**

### 隐式实例化 vs 显式实例化

```cpp
// 隐式实例化（自动）
template<typename T>
T add(T a, T b) { return a + b; }

int x = add(1, 2);  // 编译器自动实例化 add<int>

// 显式实例化（手动）
template int add<int>(int, int);  // 强制实例化

// 显式实例化声明（外部实例化，C++11）
extern template int add<int>(int, int);  // 不在此处实例化
```

### 实例化时机

```cpp
// 模板定义
template<typename T>
void foo(T x) {
    x.nonexistent_method();  // 错误，但只有实例化时才会报错
}

// 不调用，不实例化，不报错
// foo<int>(42);  // 调用才会实例化，才会报错
```

**模板代码只有在使用时才会编译检查**

---

## 5️⃣ 非类型模板参数

### 整数模板参数

```cpp
template<typename T, size_t N>
class Array {
    T data_[N];  // 固定大小数组

public:
    size_t size() const { return N; }

    T& operator[](size_t index) {
        return data_[index];
    }
};

// 使用
Array<int, 5> arr1;      // 5 个 int
Array<double, 10> arr2;  // 10 个 double

// N 是编译期常量
std::cout << arr1.size() << std::endl;  // 输出：5
```

### std::array 的实现

```cpp
// 标准库的 std::array 就是这样实现的
template<typename T, size_t N>
struct array {
    T elements[N];

    T& operator[](size_t i) { return elements[i]; }
    size_t size() const { return N; }
};

std::array<int, 5> arr = {1, 2, 3, 4, 5};
```

---

## 6️⃣ 模板特化

### 完全特化（Full Specialization）

```cpp
// 通用模板
template<typename T>
class Printer {
public:
    void print(const T& value) {
        std::cout << "通用: " << value << std::endl;
    }
};

// 特化：针对 bool 类型
template<>
class Printer<bool> {
public:
    void print(bool value) {
        std::cout << "bool: " << (value ? "true" : "false") << std::endl;
    }
};

// 使用
Printer<int> p1;
p1.print(42);  // 输出：通用: 42

Printer<bool> p2;
p2.print(true);  // 输出：bool: true
```

### 函数模板特化

```cpp
// 通用模板
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// 特化：针对 const char*
template<>
const char* max<const char*>(const char* a, const char* b) {
    return (strcmp(a, b) > 0) ? a : b;
}

// 使用
int x = max(3, 5);  // 通用版本
const char* s = max("abc", "xyz");  // 特化版本
```

### 偏特化（Partial Specialization）

```cpp
// 通用模板
template<typename T1, typename T2>
class Pair {
public:
    void print() {
        std::cout << "通用 Pair" << std::endl;
    }
};

// 偏特化：两个类型相同
template<typename T>
class Pair<T, T> {
public:
    void print() {
        std::cout << "相同类型 Pair" << std::endl;
    }
};

// 偏特化：指针类型
template<typename T>
class Pair<T*, T*> {
public:
    void print() {
        std::cout << "指针 Pair" << std::endl;
    }
};

// 使用
Pair<int, double> p1;
p1.print();  // 输出：通用 Pair

Pair<int, int> p2;
p2.print();  // 输出：相同类型 Pair

Pair<int*, int*> p3;
p3.print();  // 输出：指针 Pair
```

---

## 7️⃣ 模板与头文件

### 模板定义必须在头文件中

```cpp
// ❌ 错误做法
// my_template.h
template<typename T>
T add(T a, T b);

// my_template.cpp
template<typename T>
T add(T a, T b) {
    return a + b;
}

// main.cpp
#include "my_template.h"
int x = add(1, 2);  // 💥 链接错误：找不到 add<int> 的定义
```

**原因：**
- 模板实例化发生在编译期
- 编译器需要看到完整的模板定义才能实例化
- 如果定义在 .cpp 中，其他文件看不到，无法实例化

```cpp
// ✅ 正确做法：定义在头文件中
// my_template.h
template<typename T>
T add(T a, T b) {
    return a + b;  // 定义在头文件
}

// main.cpp
#include "my_template.h"
int x = add(1, 2);  // ✅ 编译器看到定义，可以实例化
```

### 显式实例化（例外情况）

```cpp
// my_template.h
template<typename T>
T add(T a, T b);

// my_template.cpp
template<typename T>
T add(T a, T b) {
    return a + b;
}

// 显式实例化需要的类型
template int add<int>(int, int);
template double add<double>(double, double);

// main.cpp 中只能用这些显式实例化的类型
```

---

## 8️⃣ 常见陷阱

### 陷阱 1：模板参数推导失败

```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// ❌ 错误：类型不一致
int x = max(3, 5.2);  // T 是 int 还是 double？推导失败

// ✅ 解决方案 1：显式指定
int y = max<double>(3, 5.2);  // T = double

// ✅ 解决方案 2：改成两个模板参数
template<typename T1, typename T2>
auto max(T1 a, T2 b) -> decltype(a > b ? a : b) {
    return (a > b) ? a : b;
}
```

### 陷阱 2：比较指针而不是字符串

```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

const char* s1 = "abc";
const char* s2 = "xyz";

// ❌ 错误：比较指针地址，不是字符串内容
const char* result = max(s1, s2);

// ✅ 正确：特化或用 std::string
template<>
const char* max(const char* a, const char* b) {
    return (strcmp(a, b) > 0) ? a : b;
}
```

### 陷阱 3：依赖模板参数的名字

```cpp
template<typename T>
class MyClass {
public:
    void foo() {
        // ❌ 错误：编译器不知道 T::value_type 是类型还是静态变量
        T::value_type x;

        // ✅ 正确：用 typename 告诉编译器这是类型
        typename T::value_type x;
    }
};
```

### 陷阱 4：模板代码膨胀

```cpp
template<typename T>
void process(const std::vector<T>& vec) {
    // 大量代码...
}

// 每种类型都会生成一份代码
process(std::vector<int>{});     // 生成 process<int>
process(std::vector<double>{});  // 生成 process<double>
process(std::vector<string>{});  // 生成 process<string>

// 代码体积膨胀！
```

**解决方案：提取非模板代码**

```cpp
// 非模板部分
void process_impl(void* data, size_t size, size_t elem_size) {
    // 大量代码...
}

// 模板部分（很薄）
template<typename T>
void process(const std::vector<T>& vec) {
    process_impl(vec.data(), vec.size(), sizeof(T));
}
```

---

## 9️⃣ 变参模板（Variadic Templates，C++11）

### 基本语法

```cpp
// 可以接受任意数量的参数
template<typename... Args>
void print(Args... args) {
    // ...
}

print(1, 2, 3);              // Args = int, int, int
print("hello", 42, 3.14);    // Args = const char*, int, double
```

### 递归展开

```cpp
// 递归终止
void print() {
    std::cout << std::endl;
}

// 递归展开
template<typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << first << " ";
    print(rest...);  // 递归调用
}

// 使用
print(1, 2, 3, "hello", 3.14);
// 输出：1 2 3 hello 3.14
```

### 折叠表达式（C++17）

```cpp
// 求和
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // 折叠表达式
}

int result = sum(1, 2, 3, 4, 5);  // 15

// 打印
template<typename... Args>
void print(Args... args) {
    (std::cout << ... << args) << std::endl;
}

print(1, 2, 3, "hello");
// 输出：123hello
```

---

## 🔟 SFINAE 和 Concepts（高级）

### SFINAE（Substitution Failure Is Not An Error）

```cpp
// 只对整数类型有效
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
foo(T x) {
    return x * 2;
}

foo(10);    // ✅ int 是整数
// foo(3.14);  // ❌ double 不是整数，SFINAE 排除
```

### Concepts（C++20，更清晰）

```cpp
// 定义概念
template<typename T>
concept Integral = std::is_integral_v<T>;

// 使用概念
template<Integral T>
T foo(T x) {
    return x * 2;
}

foo(10);    // ✅
// foo(3.14);  // ❌ 编译错误：不满足 Integral
```

---

## 🎯 总结

### 核心概念

```cpp
// 1. 函数模板
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// 2. 类模板
template<typename T>
class Stack {
    std::vector<T> elements_;
    // ...
};

// 3. 特化
template<>
class Stack<bool> {
    // 针对 bool 的特殊实现
};

// 4. 变参模板
template<typename... Args>
void print(Args... args) {
    (std::cout << ... << args);
}
```

### 使用原则

1. **函数模板可以自动推导类型**
   ```cpp
   auto x = max(3, 5);  // 自动推导 T = int
   ```

2. **类模板必须显式指定类型（C++17 前）**
   ```cpp
   Stack<int> s;  // 必须指定
   ```

3. **模板定义放在头文件中**
   ```cpp
   // template.h
   template<typename T>
   T add(T a, T b) { return a + b; }  // 定义在头文件
   ```

4. **用 typename 修饰依赖类型**
   ```cpp
   typename T::value_type x;
   ```

### 常见应用

```cpp
// 1. 标准库容器
std::vector<int> vec;
std::map<string, int> map;

// 2. 算法
std::sort(vec.begin(), vec.end());
std::find(vec.begin(), vec.end(), 42);

// 3. 智能指针
std::unique_ptr<int> p;
std::shared_ptr<string> sp;

// 4. 自定义泛型类
template<typename T>
class MyContainer { /* ... */ };
```

### 记住

- **模板 = 泛型编程 = 一次编写，处处复用**
- **实例化发生在编译期**
- **定义必须在头文件中**
- **C++17 折叠表达式让变参模板更简单**
- **C++20 Concepts 让模板约束更清晰**

---

## 🚀 下一步

**恭喜！阶段 1（现代 C++ 基础）全部完成！**

学完模板后，你已经掌握了：
1. ✅ 现代 C++ 思维
2. ✅ RAII 原则
3. ✅ 智能指针
4. ✅ 标准容器
5. ✅ 移动语义
6. ✅ Lambda 表达式
7. ✅ 模板基础

**接下来可以学习：**
- **阶段 2：并发编程**（线程、锁、原子操作）
- **实战项目**（应用所学知识）
- **深入主题**（高级模板、元编程）

**配套实践代码：** [practices/06_templates_basics.cpp](practices/06_templates_basics.cpp)
