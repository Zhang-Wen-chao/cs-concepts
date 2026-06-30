# 移动语义

> 现代 C++ 的性能革命：转移所有权，不拷贝数据

## 问题：拷贝很慢

```cpp
// 旧 C++：拷贝 100 万个元素（慢）
std::vector<int> create() {
    std::vector<int> vec(1000000);
    return vec;  // 拷贝？
}

// 现代 C++：移动，O(1)（快）
std::vector<int> v = create();  // 自动移动，不拷贝
```

## 左值 vs 右值

```cpp
int x = 10;
//  ↑   ↑
// 左值 右值

// 左值：有名字，可以取地址
int a = 5;
int* p = &a;  // ✅

// 右值：临时对象，不能取地址
int b = 10 + 20;  // 10 + 20 是右值
// int* p = &(10 + 20);  // ❌

// 关键：右值马上销毁，可以"偷"走资源（移动）
```

## 移动构造和移动赋值

```cpp
class MyVector {
    int* data_;
    size_t size_;
public:
    // 移动构造函数
    MyVector(MyVector&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;  // "偷"走资源，掏空原对象
        other.size_ = 0;
    }

    // 移动赋值运算符
    MyVector& operator=(MyVector&& other) noexcept {
        if (this != &other) {
            delete[] data_;  // 释放旧资源
            data_ = other.data_;  // 偷走新资源
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
};
```

**关键点**：
- `&&` 右值引用
- `noexcept` 保证不抛异常（vector 扩容时才会用移动）
- 转移资源，将源对象置空

## std::move

**std::move 不移动任何东西，只是类型转换（左值 → 右值引用）**

```cpp
std::string s1 = "hello";
std::string s2 = std::move(s1);  // 强制移动
// s1 现在是空的（被掏空）

// ⚠️ 移动后不要再用原对象
// std::cout << s1;  // 危险
```

### 什么时候用 std::move？

```cpp
// ✅ 转移所有权
std::string s2 = std::move(s1);

// ✅ 容器中移动元素
vec.push_back(std::move(s));

// ❌ 返回语句中不要用（妨碍 RVO）
// RVO = Return Value Optimization（返回值优化）
// 编译器的自动优化：直接在目标位置构造对象，零拷贝零移动
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return std::move(vec);  // ❌ 错误：破坏 RVO，退化为移动（更慢）
}

// ✅ 正确：让编译器优化
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return vec;  // 编译器自动优化（RVO 或移动）
               // RVO：直接在调用者位置构造，零开销
}
```

## 返回值自动移动（详解）

### 问题：返回大对象会拷贝吗？

```cpp
std::vector<int> create() {
    std::vector<int> v(1000000);  // 1. 创建局部变量 v
    return v;                     // 2. 返回 v，会拷贝吗？
}

auto result = create();  // 3. result 接收返回值
```

### 答案：不会拷贝，会移动（O(1)）

**旧 C++98**：
- 拷贝 100 万个 int（慢，O(n)）
- 拷贝完后，v 被销毁（浪费）

**现代 C++**：
- 不拷贝，直接移动（快，O(1)）
- 移动 = 偷走 v 的内部指针
- v 被掏空后销毁

### 移动的过程

```cpp
return v;
// 编译器识别：v 是局部变量，马上销毁，可以"偷"走资源

// 执行移动构造（底层实现）：
result.data_ = v.data_;     // 偷走指针（O(1)）
result.size_ = v.size_;
v.data_ = nullptr;          // v 被掏空
v.size_ = 0;

// v 销毁（已经是空的，不会释放内存）
// result 现在拥有那 100 万个 int
```

### 为什么是 O(1)？

```cpp
// 拷贝（O(n)，慢）
for (int i = 0; i < 1000000; i++) {
    result[i] = v[i];  // 复制 100 万次
}

// 移动（O(1)，快）
result.data_ = v.data_;  // 只是拷贝一个指针
v.data_ = nullptr;       // 置空一个指针
```

### 什么时候自动移动？

```cpp
// 1. 返回局部变量（最常见）
std::vector<int> foo() {
    std::vector<int> v(1000);
    return v;  // ✅ 自动移动
}

// 2. 返回临时对象
std::vector<int> bar() {
    return std::vector<int>(1000);  // ✅ 自动移动
}
```

### 关键：不要写 std::move

```cpp
// ❌ 错误（妨碍编译器优化）
std::vector<int> foo() {
    std::vector<int> v(1000);
    return std::move(v);  // 妨碍 RVO
}

// ✅ 正确（让编译器自动优化）
std::vector<int> foo() {
    std::vector<int> v(1000);
    return v;  // 编译器自动优化（RVO 或移动）
}
```

### 性能排序

```
RVO/NRVO（编译器优化） > 移动 > 拷贝
   O(0)               O(1)   O(n)
   零开销              偷指针   复制数据
```

## 常见陷阱

```cpp
// ❌ 使用被移动的对象
std::string s1 = "hello";
std::string s2 = std::move(s1);
std::cout << s1;  // 危险：s1 被掏空

// ❌ const 对象不能移动
const std::string s = "hello";
auto s2 = std::move(s);  // 实际是拷贝

// ❌ 返回时不要 std::move
return std::move(vec);  // 妨碍 RVO

// ❌ 移动构造要 noexcept
MyClass(MyClass&& o) { }  // 缺少 noexcept，vector 扩容时不会用移动
```

## Rule of Zero

**推荐：让标准库管理资源，编译器自动生成移动操作**

```cpp
// ✅ Rule of Zero
class Good {
    std::string name_;
    std::vector<int> data_;
    std::unique_ptr<int> ptr_;
    // 编译器自动生成移动构造和移动赋值
};

// ❌ Rule of Five（需要时）
class Manual {
    int* data_;
    // 需要手写 5 个：析构、拷贝构造、拷贝赋值、移动构造、移动赋值
};
```

## 核心要点

1. **移动 = 转移所有权，不拷贝数据**（O(1)）
2. **返回值自动移动，不需要 std::move**
3. **移动后的对象不要再用**
4. **移动构造要 `noexcept`**
5. **优先 Rule of Zero**（用标准库）
