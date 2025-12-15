# C++ 核心小抄

> 阶段 1：现代 C++ 基础 ✅

---

## 00. 现代 C++ 思维

**五大原则**：
1. RAII：构造获取，析构释放
2. 智能指针：永远不 new/delete
3. 标准容器：默认 vector
4. 移动语义：返回值自动移动
5. const 正确性：参数用 const&

**const 指针**（口诀：const 在 * 左边内容不变，右边指针不变）
```cpp
const int* p      // 指向常量（内容不可变）
int* const p      // 常量指针（指针不可变）
```

---

## 01. RAII

**核心**：资源生命周期绑定对象生命周期

```cpp
{
    std::vector<int> v(1000);  // 构造时分配
    // ...
}  // 离开作用域自动释放，即使有异常
```

**Rule of 0/3/5**：
- Rule of 0：用标准库，什么都不写
- Rule of 3：自定义析构 → 必须自定义拷贝构造、拷贝赋值
- Rule of 5：Rule of 3 + 移动构造、移动赋值

---

## 02. 智能指针

```cpp
unique_ptr  // 独占，90%情况，只能移动
shared_ptr  // 共享，引用计数，可拷贝
weak_ptr    // 打破循环引用，不增加引用计数
```

**用法**：
```cpp
auto p = std::make_unique<int>(42);
auto sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;
if (auto tmp = wp.lock()) { /* 使用 */ }
```

---

## 03. 容器

```cpp
vector           // 默认选择
unordered_map    // 键值查找 O(1)
unordered_set    // 去重 O(1)
```

**操作**：
```cpp
v.push_back(x);
m["key"] = value;
s.insert(x);
if (m.count(key)) {}
for (auto& x : c) {}
```

---

## 04. 移动语义

**六个特殊成员函数**：
```cpp
Widget w1;           // 1. 默认构造
Widget w2(w1);       // 2. 拷贝构造（= 在声明时）
w3 = w1;             // 3. 拷贝赋值（= 在赋值时）
Widget w4(move(w1)); // 4. 移动构造（创建新对象）
w4 = move(w2);       // 5. 移动赋值（已存在对象）
                     // 6. 析构
```

**引用类型**：
```cpp
T&                // 左值引用
const T&          // 万能引用（函数参数首选）
T&&               // 右值引用（移动语义）
```

**要点**：
- 移动 = 偷资源 O(1)，拷贝 = 复制数据 O(n)
- 返回值自动移动，不写 std::move
- 移动后的对象别再用
- 移动函数标记 noexcept

---

## 05. Lambda

```cpp
[捕获](参数) { 函数体 }
```

**捕获**：`[]` `[x]` `[&x]` `[=]` `[&]`

**示例**：
```cpp
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 5; });
```

---

## 06. 模板

```cpp
template<typename T>           // 函数模板（自动推导）
template<typename T> class Box // 类模板（显式指定）
template<typename... Args>     // 变长模板
```

**示例**：
```cpp
T max(T a, T b) { return a > b ? a : b; }
max(3, 5);  // 自动推导 T = int

Box<int> b(42);  // 类模板必须显式指定
```

---

## 写代码时记住

1. 永远不手动 new/delete
2. 函数参数用 const&
3. 默认用 vector
4. 移动函数标记 noexcept
5. 返回值别写 std::move
