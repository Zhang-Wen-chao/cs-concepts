# 移动语义详解

> 现代 C++ 的性能革命

## 🎯 本课目标

- 理解拷贝的性能问题
- 掌握移动语义的原理
- 理解左值和右值
- 正确使用 std::move
- 避免常见的移动语义错误

---

## 1️⃣ 问题：拷贝很慢

### 旧 C++ 的性能问题

```cpp
// 旧 C++98：拷贝大对象很慢
std::vector<int> create_large_vector() {
    std::vector<int> vec(1000000);  // 100 万个元素
    // ... 填充数据 ...
    return vec;  // 💥 拷贝 100 万个元素！（慢）
}

std::vector<int> v = create_large_vector();  // 💥 又拷贝一次！
```

**问题：**
- 拷贝大对象（vector、string 等）开销巨大
- 临时对象马上就要销毁，拷贝很浪费
- 性能损失严重

### 现代 C++ 的解决方案

```cpp
// 现代 C++：移动，不拷贝
std::vector<int> create_large_vector() {
    std::vector<int> vec(1000000);
    return vec;  // ✅ 移动，不拷贝（O(1)）
}

std::vector<int> v = create_large_vector();  // ✅ 移动，不拷贝
```

**移动语义：**
- 不拷贝数据，只转移所有权
- 像"偷"走资源，而不是"复制"资源
- 时间复杂度 O(1)，不管对象多大

---

## 2️⃣ 核心概念：左值和右值

### 什么是左值和右值？

**简单理解：**

```cpp
int x = 10;
//  ↑   ↑
// 左值 右值

// 左值（lvalue）：有名字，可以取地址
int a = 5;
int* p = &a;  // ✅ 可以取地址

// 右值（rvalue）：临时对象，没有名字
int b = 10 + 20;  // 10 + 20 是右值
// int* p = &(10 + 20);  // ❌ 不能取地址
```

**更准确的定义：**

```cpp
// 左值：表达式结束后还存在的对象
int x = 5;       // x 是左值
int y = x + 1;   // y 是左值，x + 1 是右值

// 右值：临时对象，表达式结束后就销毁
int z = foo();   // foo() 的返回值是右值（临时对象）
```

### 为什么需要区分？

```cpp
void process(std::string s);  // 参数是左值

std::string str = "hello";
process(str);           // 传左值：需要拷贝（因为 str 还要用）

process("world");       // 传右值：可以移动（"world" 是临时对象）
process(get_string());  // 传右值：可以移动（返回值是临时对象）
```

**关键：**
- 左值可能还要用，不能"偷"走资源
- 右值马上销毁，可以"偷"走资源（移动）

---

## 3️⃣ 移动构造函数和移动赋值

### 五大函数（Rule of Five）

```cpp
class MyVector {
    int* data_;
    size_t size_;

public:
    // 1. 构造函数
    MyVector(size_t size) : size_(size) {
        data_ = new int[size];
        std::cout << "构造: 分配 " << size << " 个元素" << std::endl;
    }

    // 2. 析构函数
    ~MyVector() {
        delete[] data_;
        std::cout << "析构: 释放内存" << std::endl;
    }

    // 3. 拷贝构造函数
    MyVector(const MyVector& other) : size_(other.size_) {
        data_ = new int[size_];
        std::copy(other.data_, other.data_ + size_, data_);
        std::cout << "拷贝构造: 复制 " << size_ << " 个元素" << std::endl;
    }

    // 4. 拷贝赋值运算符
    MyVector& operator=(const MyVector& other) {
        if (this != &other) {
            delete[] data_;  // 释放旧资源
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
            std::cout << "拷贝赋值: 复制 " << size_ << " 个元素" << std::endl;
        }
        return *this;
    }

    // 5. 移动构造函数（C++11）
    MyVector(MyVector&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        // "偷"走资源
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "移动构造: 转移所有权（O(1)）" << std::endl;
    }

    // 6. 移动赋值运算符（C++11）
    MyVector& operator=(MyVector&& other) noexcept {
        if (this != &other) {
            delete[] data_;  // 释放旧资源
            // "偷"走资源
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
            std::cout << "移动赋值: 转移所有权（O(1)）" << std::endl;
        }
        return *this;
    }
};
```

### 关键点

**移动构造函数：**
```cpp
MyVector(MyVector&& other) noexcept;
//                ^^                  ↑
//              右值引用            不抛异常
```

- `&&`：右值引用（绑定到右值）
- `noexcept`：保证不抛异常（重要！）
- 实现：转移资源，将源对象置空

**为什么要 `noexcept`？**

```cpp
std::vector<MyVector> vec;
vec.push_back(my_vec);

// vector 扩容时：
// - 如果移动构造 noexcept → 用移动（快）
// - 如果移动构造可能抛异常 → 用拷贝（安全但慢）
```

---

## 4️⃣ std::move：强制移动

### std::move 的作用

```cpp
#include <utility>

std::string s1 = "hello";
std::string s2 = s1;              // 拷贝
std::string s3 = std::move(s1);   // 移动

// s1 现在是空的（被"掏空"了）
```

**std::move 做了什么？**

```cpp
// 简化实现：
template<typename T>
T&& move(T& t) {
    return static_cast<T&&>(t);  // 转换成右值引用
}
```

- `std::move` 不移动任何东西！
- 只是把左值转换成右值引用
- 告诉编译器："这个对象可以被移动"

### 什么时候用 std::move？

```cpp
// ✅ 转移所有权
std::string s1 = "hello";
std::string s2 = std::move(s1);  // 转移所有权给 s2
// s1 不再使用

// ✅ 返回局部变量（某些情况）
std::unique_ptr<int> create() {
    auto p = std::make_unique<int>(42);
    return std::move(p);  // 显式移动
}

// ✅ 容器中移动元素
std::vector<std::string> v1 = {"a", "b", "c"};
std::vector<std::string> v2;
v2.push_back(std::move(v1[0]));  // 移动第一个元素

// ❌ 不要在返回语句中用 std::move（妨碍 RVO）
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return std::move(vec);  // ❌ 错误！妨碍优化
}

// ✅ 正确：让编译器优化
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return vec;  // ✅ 编译器自动优化（RVO 或移动）
}
```

---

## 5️⃣ 编译器优化：RVO 和 NRVO

### RVO（Return Value Optimization）

```cpp
std::vector<int> create_vector() {
    return std::vector<int>(1000);  // 临时对象
}

std::vector<int> v = create_vector();

// 编译器优化：
// 不拷贝，不移动，直接在 v 的位置构造！
// 零开销！
```

### NRVO（Named Return Value Optimization）

```cpp
std::vector<int> create_vector() {
    std::vector<int> vec(1000);  // 命名对象
    // ... 填充数据 ...
    return vec;
}

std::vector<int> v = create_vector();

// 编译器优化：
// 直接在 v 的位置构造 vec，不拷贝也不移动
```

### 优化等级

```
性能：RVO/NRVO > 移动 > 拷贝

RVO/NRVO（编译器优化）: O(0)（零开销）
移动:                    O(1)
拷贝:                    O(n)
```

**注意：不要手动 std::move 阻止 RVO！**

```cpp
// ❌ 错误
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return std::move(vec);  // 阻止了 RVO！
}

// ✅ 正确
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return vec;  // 编译器自动优化（RVO 或移动）
}
```

---

## 6️⃣ 移动语义的应用场景

### 场景 1：返回大对象

```cpp
// 旧代码：担心性能
std::vector<int> process_data() {
    std::vector<int> result(1000000);
    // ... 处理 ...
    return result;  // 担心拷贝？
}

// 现代 C++：不用担心
std::vector<int> result = process_data();  // RVO 或移动，很快
```

### 场景 2：转移所有权

```cpp
// unique_ptr 只能移动，不能拷贝
std::unique_ptr<int> p1 = std::make_unique<int>(42);

// std::unique_ptr<int> p2 = p1;  // ❌ 编译错误
std::unique_ptr<int> p2 = std::move(p1);  // ✅ 转移所有权

// p1 现在是空的
```

### 场景 3：容器操作

```cpp
std::vector<std::string> vec;

std::string s = "long long string";

// 拷贝
vec.push_back(s);           // s 还要用，拷贝

// 移动
vec.push_back(std::move(s)); // s 不再用，移动（更快）
```

### 场景 4：交换（swap）

```cpp
// 旧实现：三次拷贝
template<typename T>
void swap_old(T& a, T& b) {
    T temp = a;    // 拷贝
    a = b;         // 拷贝
    b = temp;      // 拷贝
}

// 新实现：三次移动
template<typename T>
void swap_new(T& a, T& b) {
    T temp = std::move(a);    // 移动
    a = std::move(b);         // 移动
    b = std::move(temp);      // 移动
}

// 标准库的 std::swap 就是这样实现的
```

---

## 7️⃣ 常见陷阱

### 陷阱 1：使用被移动的对象

```cpp
std::string s1 = "hello";
std::string s2 = std::move(s1);

std::cout << s1 << std::endl;  // ⚠️ 危险！s1 被掏空了

// s1 现在处于"有效但未指定"的状态
// 可以：赋新值、销毁
// 不可以：使用（未定义行为）
```

**规则：移动后不要使用原对象（除非重新赋值）**

```cpp
std::string s1 = "hello";
std::string s2 = std::move(s1);

// ❌ 错误
std::cout << s1.size() << std::endl;  // 未定义行为

// ✅ 正确
s1 = "new value";  // 重新赋值
std::cout << s1.size() << std::endl;  // 现在可以用了
```

### 陷阱 2：const 对象不能移动

```cpp
const std::string s1 = "hello";
std::string s2 = std::move(s1);  // ⚠️ 实际上调用的是拷贝构造

// const 对象不能修改，所以不能"掏空"
// std::move 对 const 对象无效
```

### 陷阱 3：返回语句中不要 std::move

```cpp
// ❌ 错误
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return std::move(vec);  // 妨碍 RVO
}

// ✅ 正确
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return vec;  // 编译器自动优化
}
```

### 陷阱 4：移动构造函数要 noexcept

```cpp
// ❌ 不好
class MyClass {
    MyClass(MyClass&& other) {  // 可能抛异常
        // ...
    }
};

// ✅ 好
class MyClass {
    MyClass(MyClass&& other) noexcept {  // 保证不抛异常
        // ...
    }
};

// 原因：vector 扩容时只有 noexcept 才会用移动
```

---

## 8️⃣ Rule of Zero / Rule of Five

### Rule of Zero（推荐）

**原则：不管理资源，让标准库管理**

```cpp
// ✅ 好：不需要自己写析构、拷贝、移动
class Good {
    std::string name_;
    std::vector<int> data_;
    std::unique_ptr<Resource> resource_;

    // 编译器自动生成所有特殊成员函数
    // 自动支持移动，自动正确！
};
```

### Rule of Five（自己管理资源时）

**原则：如果需要自定义析构函数，通常需要自定义所有五个**

```cpp
class MyVector {
public:
    ~MyVector();                              // 1. 析构
    MyVector(const MyVector&);                // 2. 拷贝构造
    MyVector& operator=(const MyVector&);     // 3. 拷贝赋值
    MyVector(MyVector&&) noexcept;            // 4. 移动构造
    MyVector& operator=(MyVector&&) noexcept; // 5. 移动赋值
};
```

### 推荐做法

```cpp
// ⭐⭐⭐⭐⭐ 最推荐：Rule of Zero
// 用标准库管理资源
class Recommended {
    std::vector<int> data_;
    std::unique_ptr<Resource> resource_;
    // 编译器自动生成，自动正确
};

// ⭐⭐⭐ 可以接受：Rule of Five
// 需要自己管理资源时
class Acceptable {
    int* data_;
    ~Acceptable();
    // ... 其他四个 ...
};

// ❌ 不推荐：只定义部分
class Bad {
    int* data_;
    ~Bad() { delete[] data_; }
    // 缺少拷贝/移动 → 浅拷贝 → 双重释放
};
```

---

## 9️⃣ 完美转发（Perfect Forwarding）

### 问题：参数转发

```cpp
template<typename T>
void wrapper(T arg) {
    foo(arg);  // 总是拷贝
}

std::string s = "hello";
wrapper(s);              // 拷贝
wrapper(std::move(s));   // 还是拷贝（arg 是左值）
```

### 解决方案：完美转发

```cpp
template<typename T>
void wrapper(T&& arg) {  // 万能引用（Universal Reference）
    foo(std::forward<T>(arg));  // 完美转发
}

std::string s = "hello";
wrapper(s);              // 转发左值
wrapper(std::move(s));   // 转发右值
wrapper("literal");      // 转发右值
```

**std::forward 的作用：**
- 左值 → 转发为左值
- 右值 → 转发为右值
- 保持原始的值类别

---

## 🔟 性能对比

### 测试场景

```cpp
// 拷贝 vs 移动
std::vector<std::string> vec(1000000, "long string");

// 拷贝
auto v1 = vec;  // 慢（拷贝 100 万个字符串）

// 移动
auto v2 = std::move(vec);  // 快（O(1)，只转移指针）
```

### 性能数据（示例）

```
操作             时间复杂度    典型耗时
--------------------------------------
vector 拷贝      O(n)         100 ms
vector 移动      O(1)         0.001 ms
string 拷贝      O(n)         取决于长度
string 移动      O(1)         纳秒级
unique_ptr 拷贝  不支持       -
unique_ptr 移动  O(1)         零开销
```

---

## 🎯 总结

### 核心概念

```cpp
// 1. 左值 vs 右值
int x = 5;        // x 是左值
int y = x + 1;    // x + 1 是右值

// 2. 移动语义：转移所有权，不拷贝数据
std::string s1 = "hello";
std::string s2 = std::move(s1);  // 移动，不拷贝

// 3. 移动后不能用
// s1 现在是空的，不要再用

// 4. 返回值自动优化
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return vec;  // RVO 或移动，很快
}
```

### 最佳实践

1. **让编译器自动生成**（Rule of Zero）
   ```cpp
   // 用 unique_ptr、vector 等管理资源
   class MyClass {
       std::unique_ptr<Resource> resource_;
   };
   ```

2. **移动后不要用**
   ```cpp
   auto s2 = std::move(s1);
   // 不要再用 s1
   ```

3. **返回值不要 std::move**
   ```cpp
   return vec;  // ✅ 编译器优化
   // return std::move(vec);  // ❌ 妨碍优化
   ```

4. **移动构造函数要 noexcept**
   ```cpp
   MyClass(MyClass&&) noexcept;
   ```

5. **const 对象不能移动**
   ```cpp
   const std::string s = "hello";
   auto s2 = std::move(s);  // 实际是拷贝
   ```

### 性能提升

```
操作                旧 C++     现代 C++
----------------------------------------
返回大对象          慢         快（RVO/移动）
容器扩容            慢         快（移动元素）
交换大对象          慢         快（移动）
转移所有权          不安全     安全（unique_ptr）
```

### 记住

- **移动 = 转移所有权，不拷贝数据**
- **返回值让编译器优化，不要手动 move**
- **移动后的对象不要用**
- **默认用 Rule of Zero（让标准库管理资源）**

---

## 🚀 下一步

学完移动语义后，接下来学习：
1. **Lambda 表达式**（函数式编程）
2. **模板基础**（泛型编程）
3. **完美转发**（深入理解）

**配套实践代码：** [practices/04_move_semantics.cpp](practices/04_move_semantics.cpp)
