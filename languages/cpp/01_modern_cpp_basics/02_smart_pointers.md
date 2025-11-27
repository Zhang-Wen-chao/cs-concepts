# 智能指针详解

> 告别裸指针，拥抱自动内存管理

## 🎯 本课目标

- 理解裸指针的三大问题
- 掌握三种智能指针的用法和选择
- 理解所有权（ownership）的概念
- 避免常见的智能指针陷阱

---

## 1️⃣ 裸指针的三大罪状

### 罪状 1：所有权不清晰

```cpp
void foo(int* p);  // 谁负责释放 p？
```

**问题：**
- 调用者负责 delete？
- 函数内部负责 delete？
- 不知道！容易重复释放或忘记释放

### 罪状 2：容易忘记释放

```cpp
void process() {
    int* data = new int[1000];

    // ... 100 行代码 ...

    if (error) {
        return;  // 💥 忘记 delete[]，内存泄漏
    }

    delete[] data;  // 只有正常流程才会执行
}
```

### 罪状 3：悬空指针（Dangling Pointer）

```cpp
int* p = new int(10);
delete p;
*p = 20;  // 💥 访问已释放的内存，未定义行为
```

---

## 2️⃣ 智能指针：自动管理内存

**核心思想：** 用 RAII 管理指针

```cpp
// ❌ 裸指针
int* p = new int(10);
delete p;  // 容易忘记

// ✅ 智能指针
std::unique_ptr<int> p(new int(10));
// 自动 delete，不会忘记
```

**三种智能指针：**
1. `unique_ptr` - 独占所有权（最常用，推荐！）
2. `shared_ptr` - 共享所有权（需要共享时使用）
3. `weak_ptr` - 弱引用（打破循环引用）

---

## 3️⃣ unique_ptr：独占所有权

### 核心特点

```cpp
std::unique_ptr<int> p1(new int(10));
// p1 独占这块内存

// std::unique_ptr<int> p2 = p1;  // ❌ 编译错误：不能拷贝
std::unique_ptr<int> p2 = std::move(p1);  // ✅ 可以移动（转移所有权）

// p1 现在是空的，p2 拥有内存
```

**所有权语义：**
- 一个 unique_ptr 独占一块内存
- 不能拷贝，只能移动
- 所有权清晰：谁拥有，谁负责释放

### 基本用法

```cpp
#include <memory>

// 1. 创建 unique_ptr（C++14 推荐方式）
auto p1 = std::make_unique<int>(42);

// 2. 旧方式（也可以）
std::unique_ptr<int> p2(new int(42));

// 3. 访问数据
std::cout << *p1 << std::endl;  // 解引用

// 4. 获取原始指针
int* raw = p1.get();

// 5. 释放所有权（返回原始指针）
int* raw2 = p1.release();  // p1 变成空的，需要手动 delete raw2

// 6. 重置
p2.reset();  // 释放内存，p2 变成空的
p2.reset(new int(100));  // 释放旧内存，指向新内存
```

### 数组版本

```cpp
// 动态数组
std::unique_ptr<int[]> arr(new int[100]);
arr[0] = 42;  // 可以用下标访问

// 或者用 make_unique（C++14）
auto arr2 = std::make_unique<int[]>(100);

// ⚠️ 注意：析构时会调用 delete[]（不是 delete）
```

### 自定义删除器

```cpp
// 管理 FILE*
auto file_deleter = [](FILE* f) {
    if (f) std::fclose(f);
};

std::unique_ptr<FILE, decltype(file_deleter)> file(
    std::fopen("data.txt", "r"),
    file_deleter
);

// 或者更简单（C++17）
std::unique_ptr<FILE, void(*)(FILE*)> file2(
    std::fopen("data.txt", "r"),
    [](FILE* f) { if (f) std::fclose(f); }
);
```

### 什么时候用 unique_ptr？

**答案：90% 的情况都用它！**

```cpp
// ✅ 动态分配单个对象
auto p = std::make_unique<MyClass>(args);

// ✅ 工厂函数返回值
std::unique_ptr<Base> create_object() {
    return std::make_unique<Derived>();
}

// ✅ 类的成员变量（管理资源）
class MyClass {
    std::unique_ptr<Resource> resource_;
};

// ✅ 容器中存储多态对象
std::vector<std::unique_ptr<Base>> objects;
objects.push_back(std::make_unique<Derived>());
```

---

## 4️⃣ shared_ptr：共享所有权

### 核心特点

```cpp
std::shared_ptr<int> p1 = std::make_shared<int>(42);
std::shared_ptr<int> p2 = p1;  // ✅ 可以拷贝，引用计数 +1

std::cout << p1.use_count() << std::endl;  // 输出: 2

// p2 销毁，引用计数 -1
// p1 销毁，引用计数 -1 → 0，释放内存
```

**引用计数：**
- 每次拷贝，引用计数 +1
- 每次销毁，引用计数 -1
- 引用计数变成 0 时，自动释放内存

### 基本用法

```cpp
#include <memory>

// 1. 创建 shared_ptr（推荐方式）
auto p1 = std::make_shared<int>(42);

// 2. 旧方式（不推荐，效率低）
std::shared_ptr<int> p2(new int(42));

// 3. 拷贝（共享所有权）
auto p3 = p1;  // 引用计数 +1

// 4. 查询引用计数
std::cout << p1.use_count() << std::endl;

// 5. 检查是否唯一
if (p1.unique()) {
    std::cout << "只有我一个引用" << std::endl;
}

// 6. 重置
p1.reset();  // 引用计数 -1，如果变成 0 就释放内存
```

### make_shared vs new

```cpp
// ❌ 不推荐（两次内存分配）
std::shared_ptr<int> p1(new int(42));
// 1. new 分配对象内存
// 2. shared_ptr 分配控制块内存

// ✅ 推荐（一次内存分配）
auto p2 = std::make_shared<int>(42);
// 一次分配：对象 + 控制块
```

### 什么时候用 shared_ptr？

**只在需要共享所有权时使用：**

```cpp
// ✅ 多个对象需要共享同一资源
class Node {
    std::shared_ptr<Data> shared_data_;  // 多个节点共享数据
};

// ✅ 回调函数需要保持对象存活
void async_operation(std::shared_ptr<Object> obj) {
    // obj 在异步操作期间保持存活
}

// ✅ 缓存
std::unordered_map<std::string, std::shared_ptr<Resource>> cache;

// ❌ 不需要共享时，用 unique_ptr
// 不要为了"方便拷贝"而用 shared_ptr
```

---

## 5️⃣ weak_ptr：打破循环引用

### 循环引用问题

```cpp
class Node {
public:
    std::shared_ptr<Node> next;  // 指向下一个节点
    std::shared_ptr<Node> prev;  // 指向前一个节点
};

auto n1 = std::make_shared<Node>();
auto n2 = std::make_shared<Node>();

n1->next = n2;  // n1 → n2
n2->prev = n1;  // n2 → n1

// 💥 循环引用！
// n1 的引用计数 = 2（n1 本身 + n2->prev）
// n2 的引用计数 = 2（n2 本身 + n1->next）
// 都不会变成 0，内存泄漏！
```

### weak_ptr 解决方案

```cpp
class Node {
public:
    std::shared_ptr<Node> next;  // 强引用
    std::weak_ptr<Node> prev;    // 弱引用（不增加引用计数）
};

auto n1 = std::make_shared<Node>();
auto n2 = std::make_shared<Node>();

n1->next = n2;  // n1 → n2（强引用）
n2->prev = n1;  // n2 ⇢ n1（弱引用）

// ✅ 没有循环引用
// n1 引用计数 = 1
// n2 引用计数 = 2（n2 本身 + n1->next）
// 当 n1、n2 离开作用域，都会被正确释放
```

### weak_ptr 基本用法

```cpp
auto sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;  // 弱引用，不增加引用计数

std::cout << sp.use_count() << std::endl;  // 输出: 1（不是 2）

// 使用 weak_ptr 的值：先转换成 shared_ptr
if (auto temp_sp = wp.lock()) {  // lock() 返回 shared_ptr
    std::cout << *temp_sp << std::endl;  // 安全访问
} else {
    std::cout << "对象已被释放" << std::endl;
}

// 检查对象是否还存活
if (wp.expired()) {
    std::cout << "对象已被释放" << std::endl;
}
```

### 什么时候用 weak_ptr？

```cpp
// ✅ 打破循环引用（树、图结构）
class Node {
    std::shared_ptr<Node> left, right;  // 子节点：强引用
    std::weak_ptr<Node> parent;          // 父节点：弱引用
};

// ✅ 观察者模式（观察者不拥有被观察对象）
class Observable {
    std::vector<std::weak_ptr<Observer>> observers_;
};

// ✅ 缓存（缓存不阻止对象释放）
std::unordered_map<Key, std::weak_ptr<Value>> cache;
```

---

## 6️⃣ 智能指针对比表

| 特性 | unique_ptr | shared_ptr | weak_ptr |
|------|-----------|-----------|----------|
| **所有权** | 独占 | 共享 | 不拥有 |
| **可拷贝** | ❌ | ✅ | ✅ |
| **可移动** | ✅ | ✅ | ✅ |
| **引用计数** | 无 | 有 | 不增加计数 |
| **开销** | 最小 | 中等 | 小 |
| **使用场景** | 90% 的情况 | 需要共享 | 打破循环引用 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 7️⃣ 常见陷阱和最佳实践

### 陷阱 1：不要用同一个裸指针初始化多个智能指针

```cpp
// ❌ 危险
int* raw = new int(42);
std::unique_ptr<int> p1(raw);
std::unique_ptr<int> p2(raw);  // 💥 重复释放！

// ✅ 正确
auto p1 = std::make_unique<int>(42);
// 不需要 raw 指针
```

### 陷阱 2：不要从智能指针获取裸指针后再创建智能指针

```cpp
auto p1 = std::make_unique<int>(42);
int* raw = p1.get();

// ❌ 危险
std::unique_ptr<int> p2(raw);  // 💥 重复释放
```

### 陷阱 3：shared_ptr 的循环引用

```cpp
// ❌ 循环引用
struct Node {
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev;  // 应该用 weak_ptr
};

// ✅ 正确
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;
};
```

### 陷阱 4：在函数参数中滥用智能指针

```cpp
// ❌ 不好（传值，引用计数操作有开销）
void foo(std::shared_ptr<Widget> w);

// ✅ 好（传引用）
void foo(const std::shared_ptr<Widget>& w);

// ✅ 更好（如果不需要所有权，传裸指针或引用）
void foo(Widget* w);
void foo(Widget& w);
```

### 最佳实践总结

```cpp
// 1️⃣ 默认用 unique_ptr
auto p = std::make_unique<T>(args);

// 2️⃣ 需要共享时才用 shared_ptr
auto sp = std::make_shared<T>(args);

// 3️⃣ 打破循环引用用 weak_ptr
std::weak_ptr<T> wp = sp;

// 4️⃣ 用 make_unique / make_shared（不要手动 new）
// ✅ auto p = std::make_unique<T>();
// ❌ std::unique_ptr<T> p(new T());

// 5️⃣ 函数参数按需传递
void use_only(Widget& w);           // 只使用，不拥有
void take_ownership(std::unique_ptr<Widget> w);  // 转移所有权
void share_ownership(std::shared_ptr<Widget> w); // 共享所有权
```

---

## 8️⃣ 实战示例

### 示例 1：工厂模式

```cpp
class Base {
public:
    virtual ~Base() = default;
    virtual void do_something() = 0;
};

class Derived : public Base {
public:
    void do_something() override { /* ... */ }
};

// 工厂函数返回 unique_ptr
std::unique_ptr<Base> create_object(int type) {
    if (type == 1) {
        return std::make_unique<Derived>();
    }
    return nullptr;
}

// 使用
auto obj = create_object(1);
if (obj) {
    obj->do_something();
}
```

### 示例 2：PIMPL 惯用法

```cpp
// Widget.h
class Widget {
public:
    Widget();
    ~Widget();
    void do_something();

private:
    class Impl;  // 前向声明
    std::unique_ptr<Impl> pimpl_;  // 指向实现
};

// Widget.cpp
class Widget::Impl {
public:
    void do_something_impl() { /* 实现细节 */ }
    // 私有成员...
};

Widget::Widget() : pimpl_(std::make_unique<Impl>()) {}
Widget::~Widget() = default;  // 必须在 cpp 中定义

void Widget::do_something() {
    pimpl_->do_something_impl();
}
```

### 示例 3：容器中存储多态对象

```cpp
std::vector<std::unique_ptr<Base>> objects;

objects.push_back(std::make_unique<Derived1>());
objects.push_back(std::make_unique<Derived2>());

for (auto& obj : objects) {
    obj->do_something();  // 多态调用
}
```

---

## 9️⃣ 性能考虑

### unique_ptr 的开销

```cpp
sizeof(std::unique_ptr<int>) == sizeof(int*)  // true
// 零开销！和裸指针一样大
```

**结论：** unique_ptr 零运行时开销，没理由不用！

### shared_ptr 的开销

```cpp
sizeof(std::shared_ptr<int>) == 2 * sizeof(int*)  // true
// 包含：指向对象的指针 + 指向控制块的指针

// 控制块包含：
// - 引用计数（强引用）
// - 弱引用计数
// - 删除器
```

**结论：** shared_ptr 有开销，只在需要共享时使用。

---

## 🎯 总结

### 选择指南

```
需要动态内存？
    ↓
独占所有权？ → unique_ptr（90% 的情况）
    ↓ 否
需要共享？ → shared_ptr
    ↓
有循环引用？ → 用 weak_ptr 打破
```

### 核心原则

1. **默认用 unique_ptr**
2. **需要共享才用 shared_ptr**
3. **打破循环引用用 weak_ptr**
4. **永远不要手动 delete**
5. **用 make_unique / make_shared**

### 记住

```cpp
// ❌ 旧代码
Widget* w = new Widget();
delete w;

// ✅ 现代代码
auto w = std::make_unique<Widget>();
// 自动释放，永远不会泄漏
```

---

## 🚀 下一步

学完智能指针后，接下来学习：
1. **标准容器**（vector、map、set 等）
2. **移动语义**（深入理解所有权转移）
3. **Lambda 表达式**

**配套实践代码：** [practices/02_smart_pointers.cpp](practices/02_smart_pointers.cpp)
