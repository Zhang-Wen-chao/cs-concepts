# 标准容器详解

> 不要重复造轮子，标准库已经优化到极致

## 🎯 本课目标

- 理解各种容器的特点和使用场景
- 掌握容器选择的原则
- 了解容器的性能特征
- 避免常见的容器使用错误

---

## 1️⃣ 容器分类

### 三大类容器

**顺序容器（Sequence Containers）：**
- `vector` - 动态数组（最常用）
- `deque` - 双端队列
- `list` - 双向链表
- `array` - 固定大小数组（C++11）
- `forward_list` - 单向链表（C++11）

**关联容器（Associative Containers）：**
- `set` / `multiset` - 有序集合（红黑树）
- `map` / `multimap` - 有序键值对（红黑树）

**无序关联容器（Unordered Associative Containers，C++11）：**
- `unordered_set` / `unordered_multiset` - 哈希集合
- `unordered_map` / `unordered_multimap` - 哈希表

**容器适配器（Container Adapters）：**
- `stack` - 栈（LIFO）
- `queue` - 队列（FIFO）
- `priority_queue` - 优先队列（堆）

---

## 2️⃣ vector：最常用的容器

### 核心特点

```cpp
std::vector<int> vec;  // 动态数组

// 特点：
// ✅ 连续内存，缓存友好
// ✅ 随机访问 O(1)
// ✅ 末尾插入/删除 O(1)（均摊）
// ❌ 中间插入/删除 O(n)
```

### 基本用法

```cpp
#include <vector>

// 1. 创建
std::vector<int> v1;                    // 空
std::vector<int> v2(10);                // 10 个元素，值为 0
std::vector<int> v3(10, 42);            // 10 个元素，值为 42
std::vector<int> v4 = {1, 2, 3, 4, 5};  // 初始化列表

// 2. 添加元素
v1.push_back(10);     // 末尾添加
v1.emplace_back(20);  // 原地构造（更高效）

// 3. 访问元素
int x = v4[0];        // 下标访问（不检查边界）
int y = v4.at(0);     // at() 访问（检查边界，抛异常）
int z = v4.front();   // 第一个元素
int w = v4.back();    // 最后一个元素

// 4. 大小和容量
size_t size = v4.size();      // 元素个数
size_t cap = v4.capacity();   // 容量（分配的内存）
bool empty = v4.empty();      // 是否为空

// 5. 修改
v4.pop_back();        // 删除末尾元素
v4.clear();           // 清空所有元素
v4.resize(100);       // 改变大小
v4.reserve(1000);     // 预留容量（避免重新分配）

// 6. 迭代
for (auto it = v4.begin(); it != v4.end(); ++it) {
    std::cout << *it << " ";
}

// 或者用范围 for（推荐）
for (const auto& elem : v4) {
    std::cout << elem << " ";
}
```

### 容量管理

```cpp
std::vector<int> vec;

// size vs capacity
std::cout << "size: " << vec.size() << std::endl;       // 0
std::cout << "capacity: " << vec.capacity() << std::endl; // 0

vec.push_back(1);
// size: 1, capacity: 可能是 1

vec.push_back(2);
// size: 2, capacity: 可能是 2

vec.push_back(3);
// size: 3, capacity: 可能是 4（增长策略：通常是 2 倍增长）

// 预留容量（避免多次重新分配）
vec.reserve(1000);  // capacity 变成 1000，但 size 不变
```

### push_back vs emplace_back

```cpp
class Point {
public:
    int x, y;
    Point(int x, int y) : x(x), y(y) {
        std::cout << "Point(" << x << ", " << y << ")" << std::endl;
    }
};

std::vector<Point> points;

// push_back：先构造临时对象，再拷贝/移动
points.push_back(Point(1, 2));  // 构造 Point(1,2)，然后移动

// emplace_back：直接在容器中构造（更高效）
points.emplace_back(3, 4);      // 直接在 vector 中构造 Point(3,4)
```

**结论：优先用 `emplace_back`**

### 什么时候用 vector？

**答案：90% 的情况！**

```cpp
// ✅ 需要动态数组
std::vector<int> numbers;

// ✅ 需要随机访问
int x = numbers[100];

// ✅ 需要连续内存（性能）
// vector 的缓存局部性很好

// ✅ 大部分操作在末尾
numbers.push_back(42);
numbers.pop_back();

// ❌ 大量中间插入/删除 → 用 list 或 deque
```

---

## 3️⃣ deque：双端队列

### 核心特点

```cpp
std::deque<int> dq;

// 特点：
// ✅ 两端插入/删除 O(1)
// ✅ 随机访问 O(1)
// ❌ 内存不连续（不如 vector 缓存友好）
// ❌ 迭代器可能失效
```

### 基本用法

```cpp
#include <deque>

std::deque<int> dq = {1, 2, 3, 4, 5};

// 两端操作
dq.push_front(0);   // 头部插入
dq.push_back(6);    // 尾部插入
dq.pop_front();     // 头部删除
dq.pop_back();      // 尾部删除

// 随机访问（和 vector 一样）
int x = dq[2];
```

### 什么时候用 deque？

```cpp
// ✅ 需要两端操作
std::deque<int> dq;
dq.push_front(1);  // vector 不支持
dq.push_back(2);

// ✅ 实现队列（queue）
std::queue<int> q;  // 默认底层用 deque

// ❌ 需要极致性能 → 用 vector
```

---

## 4️⃣ list：双向链表

### 核心特点

```cpp
std::list<int> lst;

// 特点：
// ✅ 任意位置插入/删除 O(1)（如果有迭代器）
// ✅ 迭代器不会失效（除了被删除的）
// ❌ 不支持随机访问
// ❌ 内存不连续，缓存不友好
// ❌ 额外的指针开销
```

### 基本用法

```cpp
#include <list>

std::list<int> lst = {1, 2, 3, 4, 5};

// 两端操作
lst.push_front(0);
lst.push_back(6);
lst.pop_front();
lst.pop_back();

// 中间插入/删除
auto it = lst.begin();
++it;  // 指向第二个元素
lst.insert(it, 99);  // 在第二个元素前插入 99
lst.erase(it);       // 删除第二个元素

// ❌ 不支持随机访问
// int x = lst[2];  // 编译错误
```

### 什么时候用 list？

```cpp
// ✅ 大量中间插入/删除
std::list<int> lst;
auto it = /* ... */;
lst.insert(it, 42);  // O(1)

// ✅ 需要迭代器稳定性
// （插入/删除不会让其他迭代器失效）

// ❌ 需要随机访问 → 用 vector
// ❌ 绝大部分情况 → 用 vector
```

**重要：list 很少用到！vector 通常更快（即使有插入/删除）**

---

## 5️⃣ map：有序键值对

### 核心特点

```cpp
std::map<std::string, int> m;

// 特点：
// ✅ 键有序（红黑树实现）
// ✅ 查找、插入、删除 O(log n)
// ❌ 比 unordered_map 慢
```

### 基本用法

```cpp
#include <map>

std::map<std::string, int> age_map;

// 1. 插入
age_map["Alice"] = 25;        // 下标插入
age_map["Bob"] = 30;
age_map.insert({"Charlie", 35});  // insert

// 2. 查找
if (age_map.count("Alice")) {
    std::cout << "Alice 存在" << std::endl;
}

auto it = age_map.find("Bob");
if (it != age_map.end()) {
    std::cout << "Bob 的年龄: " << it->second << std::endl;
}

// 3. 访问
int age = age_map["Alice"];   // 如果不存在，会插入默认值
int age2 = age_map.at("Bob"); // 如果不存在，抛异常

// 4. 遍历（按键的顺序）
for (const auto& [name, age] : age_map) {  // C++17 结构化绑定
    std::cout << name << ": " << age << std::endl;
}
// 输出：Alice: 25, Bob: 30, Charlie: 35（按字典序）

// 5. 删除
age_map.erase("Alice");
```

### 什么时候用 map？

```cpp
// ✅ 需要键有序
std::map<int, std::string> sorted_map;
for (const auto& [key, value] : sorted_map) {
    // 按 key 从小到大遍历
}

// ✅ 需要范围查询
auto it1 = sorted_map.lower_bound(10);  // >= 10 的第一个
auto it2 = sorted_map.upper_bound(20);  // > 20 的第一个

// ❌ 不需要有序 → 用 unordered_map（更快）
```

---

## 6️⃣ unordered_map：哈希表（最常用的映射）

### 核心特点

```cpp
std::unordered_map<std::string, int> m;

// 特点：
// ✅ 查找、插入、删除 O(1)（平均）
// ✅ 比 map 快
// ❌ 键无序
// ❌ 最坏情况 O(n)（哈希冲突）
```

### 基本用法

```cpp
#include <unordered_map>

std::unordered_map<std::string, int> word_count;

// 统计单词频率
std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana"};
for (const auto& word : words) {
    word_count[word]++;  // 自动初始化为 0
}

// 遍历（无序）
for (const auto& [word, count] : word_count) {
    std::cout << word << ": " << count << std::endl;
}
```

### 什么时候用 unordered_map？

**答案：大部分情况！**

```cpp
// ✅ 需要快速查找（O(1)）
std::unordered_map<int, std::string> id_to_name;

// ✅ 不需要有序
// （大部分情况都不需要有序）

// ❌ 需要有序 → 用 map
// ❌ 键不可哈希 → 用 map
```

---

## 7️⃣ set：集合

### map vs set

```cpp
// set：只存键，没有值
std::set<int> s = {1, 2, 3};

// map：存键值对
std::map<int, std::string> m = {{1, "one"}, {2, "two"}};

// unordered_set：无序集合（哈希）
std::unordered_set<int> us = {1, 2, 3};
```

### 基本用法

```cpp
#include <set>
#include <unordered_set>

// 有序集合
std::set<int> s = {3, 1, 4, 1, 5};  // 自动去重、排序
// 结果：{1, 3, 4, 5}

// 插入
s.insert(2);

// 查找
if (s.count(3)) {
    std::cout << "3 存在" << std::endl;
}

// 删除
s.erase(1);

// 遍历（有序）
for (int x : s) {
    std::cout << x << " ";  // 输出：2 3 4 5
}

// 无序集合（更快）
std::unordered_set<int> us = {3, 1, 4, 1, 5};
// 结果：{1, 3, 4, 5}（去重，但无序）
```

### 什么时候用 set？

```cpp
// ✅ 需要去重
std::set<int> unique_nums = {1, 2, 2, 3, 3, 3};
// 结果：{1, 2, 3}

// ✅ 需要快速判断存在性
if (unique_nums.count(2)) { /* ... */ }

// ✅ 需要有序 → set
// ✅ 不需要有序 → unordered_set（更快）
```

---

## 8️⃣ 容器适配器

### stack（栈）

```cpp
#include <stack>

std::stack<int> stk;

stk.push(1);
stk.push(2);
stk.push(3);

std::cout << stk.top() << std::endl;  // 3（栈顶）
stk.pop();  // 弹出 3

std::cout << stk.size() << std::endl;  // 2
```

### queue（队列）

```cpp
#include <queue>

std::queue<int> q;

q.push(1);
q.push(2);
q.push(3);

std::cout << q.front() << std::endl;  // 1（队首）
std::cout << q.back() << std::endl;   // 3（队尾）
q.pop();  // 弹出 1
```

### priority_queue（优先队列 / 堆）

```cpp
#include <queue>

// 默认：大顶堆
std::priority_queue<int> max_heap;
max_heap.push(3);
max_heap.push(1);
max_heap.push(4);

std::cout << max_heap.top() << std::endl;  // 4（最大值）
max_heap.pop();

// 小顶堆
std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;
min_heap.push(3);
min_heap.push(1);
min_heap.push(4);

std::cout << min_heap.top() << std::endl;  // 1（最小值）
```

---

## 9️⃣ 容器选择指南

### 决策树

```
需要什么容器？
    ↓
需要键值对？
    ├─ 是 → 需要有序？
    │        ├─ 是 → map
    │        └─ 否 → unordered_map（推荐）
    │
    └─ 否 → 需要去重？
             ├─ 是 → 需要有序？
             │        ├─ 是 → set
             │        └─ 否 → unordered_set（推荐）
             │
             └─ 否 → 需要什么操作？
                      ├─ 两端操作 → deque
                      ├─ 中间插入/删除 → list（罕见）
                      ├─ LIFO → stack
                      ├─ FIFO → queue
                      ├─ 优先级 → priority_queue
                      └─ 其他 → vector（默认选择）
```

### 性能对比表

| 容器 | 随机访问 | 插入/删除（头） | 插入/删除（尾） | 插入/删除（中间） | 查找 |
|------|---------|----------------|----------------|-----------------|------|
| **vector** | O(1) | O(n) | O(1) | O(n) | O(n) |
| **deque** | O(1) | O(1) | O(1) | O(n) | O(n) |
| **list** | O(n) | O(1) | O(1) | O(1)* | O(n) |
| **map** | - | - | - | - | O(log n) |
| **unordered_map** | - | - | - | - | O(1) |

*需要已有迭代器

### 推荐使用频率

```cpp
// ⭐⭐⭐⭐⭐ 最常用（90%）
std::vector<T>
std::unordered_map<K, V>
std::unordered_set<T>
std::string

// ⭐⭐⭐ 常用
std::map<K, V>  // 需要有序时
std::set<T>     // 需要有序时
std::queue<T>
std::stack<T>
std::priority_queue<T>

// ⭐⭐ 偶尔用
std::deque<T>

// ⭐ 很少用
std::list<T>
std::forward_list<T>
```

---

## 🔟 常见陷阱

### 陷阱 1：vector 的 [] 不检查边界

```cpp
std::vector<int> vec = {1, 2, 3};

// ❌ 危险（未定义行为）
int x = vec[10];  // 越界，但不报错

// ✅ 安全（抛异常）
int y = vec.at(10);  // 抛出 std::out_of_range
```

### 陷阱 2：map 的 [] 会插入元素

```cpp
std::map<std::string, int> m;

// ❌ 意外插入
int age = m["Alice"];  // 如果不存在，会插入 {"Alice", 0}

// ✅ 正确查找
auto it = m.find("Alice");
if (it != m.end()) {
    int age = it->second;
}

// 或者用 at()（C++11）
try {
    int age = m.at("Alice");  // 不存在会抛异常
} catch (const std::out_of_range& e) {
    // ...
}
```

### 陷阱 3：迭代器失效

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};

// ❌ 危险
for (auto it = vec.begin(); it != vec.end(); ++it) {
    if (*it == 3) {
        vec.erase(it);  // it 失效！
    }
}

// ✅ 正确
for (auto it = vec.begin(); it != vec.end(); ) {
    if (*it == 3) {
        it = vec.erase(it);  // erase 返回下一个有效迭代器
    } else {
        ++it;
    }
}

// ✅ 更简单（C++20）
std::erase(vec, 3);  // 直接删除所有值为 3 的元素
```

### 陷阱 4：不必要的拷贝

```cpp
std::vector<std::string> vec = {"long string 1", "long string 2"};

// ❌ 拷贝（慢）
for (auto str : vec) {
    std::cout << str << std::endl;
}

// ✅ 引用（快）
for (const auto& str : vec) {
    std::cout << str << std::endl;
}
```

---

## 1️⃣1️⃣ 最佳实践

### 1. 默认用 vector

```cpp
// ✅ 90% 的情况
std::vector<int> data;

// 只在有明确理由时才用其他容器
```

### 2. 预留容量

```cpp
std::vector<int> vec;

// ❌ 多次重新分配（慢）
for (int i = 0; i < 1000; ++i) {
    vec.push_back(i);
}

// ✅ 一次分配（快）
std::vector<int> vec2;
vec2.reserve(1000);
for (int i = 0; i < 1000; ++i) {
    vec2.push_back(i);
}
```

### 3. 用 emplace 而不是 push

```cpp
std::vector<std::pair<int, std::string>> vec;

// ❌ 构造临时对象
vec.push_back(std::make_pair(1, "one"));

// ✅ 原地构造
vec.emplace_back(1, "one");
```

### 4. 范围 for 用 const 引用

```cpp
std::vector<std::string> vec = {/* ... */};

// ✅ 只读
for (const auto& s : vec) {
    std::cout << s << std::endl;
}

// ✅ 修改
for (auto& s : vec) {
    s += " modified";
}
```

### 5. 算法优于手写循环

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};

// ❌ 手写循环
int sum = 0;
for (int x : vec) {
    sum += x;
}

// ✅ 用算法（更清晰）
#include <numeric>
int sum = std::accumulate(vec.begin(), vec.end(), 0);

// ✅ 查找
auto it = std::find(vec.begin(), vec.end(), 3);

// ✅ 排序
std::sort(vec.begin(), vec.end());
```

---

## 🎯 总结

### 容器选择原则

1. **默认用 vector**
2. 需要键值对 → **unordered_map**
3. 需要去重 → **unordered_set**
4. 需要有序 → **map / set**
5. 特殊需求 → **deque / queue / stack / priority_queue**

### 核心要点

```cpp
// 1️⃣ vector 是默认选择
std::vector<int> vec;

// 2️⃣ unordered_map > map（大部分情况）
std::unordered_map<std::string, int> m;

// 3️⃣ 预留容量
vec.reserve(1000);

// 4️⃣ emplace > push
vec.emplace_back(42);

// 5️⃣ 范围 for 用 const 引用
for (const auto& elem : vec) { /* ... */ }

// 6️⃣ 用标准算法
std::sort(vec.begin(), vec.end());
```

### 记住

- **vector：90% 的情况**
- **unordered_map：需要键值对时**
- **不要过早优化**：先用 vector，有问题再换

---

## 🚀 下一步

学完容器后，接下来学习：
1. **移动语义**（理解容器的性能优化）
2. **Lambda 表达式**（配合算法使用）
3. **迭代器**（深入理解容器遍历）

**配套实践代码：** [practices/03_containers.cpp](practices/03_containers.cpp)
