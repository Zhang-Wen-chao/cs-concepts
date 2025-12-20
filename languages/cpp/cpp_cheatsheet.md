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
- 构造时获取资源，析构时释放资源
- C++ 保证：离开作用域必调用析构，即使有异常

**Rule of 0/3/5**：
- **Rule of 0**：用标准库，什么都不写（推荐）✅
- **Rule of 5**：自己管理资源时，必须定义 5 个函数
  ```cpp
  ~T();                      // 1. 析构
  T(const T&);               // 2. 拷贝构造
  T& operator=(const T&);    // 3. 拷贝赋值
  T(T&&) noexcept;           // 4. 移动构造
  T& operator=(T&&) noexcept;// 5. 移动赋值
  ```
- **Rule of 3**：C++11 前的旧规则，已过时

**禁止拷贝的方式**：
```cpp
T(const T&) = delete;
T& operator=(const T&) = delete;
```

**noexcept 关键点**：
- `noexcept` 是程序员的承诺，不是编译器检查
- 违反承诺 → 运行时 `std::terminate`，程序崩溃
- 移动函数必须 `noexcept`，否则 vector 扩容时退化为拷贝
- 析构函数默认就是 `noexcept`，绝对不能抛异常

**RAII 类型**：
- 内存：`unique_ptr`, `shared_ptr`, `vector`, `string`
- 文件：`ifstream`, `ofstream`, `fstream`
- 锁：`lock_guard`, `unique_lock`, `scoped_lock`
- 线程：`thread`, `jthread`(C++20)

---

## 02. 智能指针

**三种类型**：
```cpp
unique_ptr  // 独占所有权，90%情况，只能移动，零开销
shared_ptr  // 共享所有权，引用计数，可拷贝
weak_ptr    // 不拥有，不增加引用计数，打破循环引用
```

**创建方式**：
```cpp
auto p = std::make_unique<int>(42);   // unique_ptr（推荐）
auto sp = std::make_shared<int>(42);  // shared_ptr（推荐）
auto arr = std::make_unique<int[]>(100); // 数组

// ❌ 不推荐
std::unique_ptr<int> p(new int(42));  // 不如 make_unique 安全
std::shared_ptr<int> sp(new int(42)); // 两次内存分配，慢
```

**unique_ptr**：
```cpp
auto p2 = std::move(p1);  // 移动所有权，p1 变空
int* raw = p.get();       // 获取原始指针（不转移所有权）
```

**shared_ptr**：
```cpp
auto p2 = p1;              // 拷贝，引用计数 +1
p1.use_count();            // 查询引用计数
p1.reset();                // 引用计数 -1
```

**weak_ptr（打破循环引用）**：
```cpp
// ❌ 循环引用会内存泄漏
struct Node {
    std::shared_ptr<Node> next;  // 强引用
    std::shared_ptr<Node> prev;  // 也强引用 💥 循环了
};

// ✅ 用 weak_ptr 打破循环
struct Node {
    std::shared_ptr<Node> next;  // 强引用
    std::weak_ptr<Node> prev;    // 弱引用 ✅
};

// 使用 weak_ptr
std::weak_ptr<int> wp = sp;  // 不增加引用计数
if (auto tmp = wp.lock()) {  // lock() 转为 shared_ptr
    // 使用 tmp
}
```

**双向引用规则**：
- 拥有方 → 被拥有方：`shared_ptr`
- 被拥有方 → 拥有方：`weak_ptr`
- 例子：父拥有子（`shared_ptr`），子引用父（`weak_ptr`）

**常见陷阱**：
```cpp
// ❌ 同一个裸指针初始化多个智能指针（重复释放）
int* raw = new int(42);
std::unique_ptr<int> p1(raw);
std::unique_ptr<int> p2(raw);  // 💥

// ❌ 从 get() 再创建智能指针（重复释放）
auto p1 = std::make_unique<int>(42);
std::unique_ptr<int> p2(p1.get());  // 💥
```

**函数参数传递**：
```cpp
void use_only(T& obj);                    // 只使用，不关心所有权
void take_ownership(std::unique_ptr<T> p); // 转移所有权
void share(std::shared_ptr<T> p);          // 共享所有权
void observe(const std::shared_ptr<T>& p); // 不改变引用计数
```

---

## 03. 容器

**选择指南**（90% 情况）：
```cpp
vector           // 默认选择（顺序存储，随机访问）
unordered_map    // 键值查找 O(1)
unordered_set    // 去重 O(1)
```

**决策树**：
```
需要键值对？
  是 → unordered_map
  否 → 需要去重？
         是 → unordered_set
         否 → vector（默认）
```

**vector（默认选择）**：
```cpp
std::vector<int> v = {1, 2, 3};
v.push_back(4);          // 末尾添加（构造临时对象再移动）
v.emplace_back(5);       // 末尾原地构造（更快，避免移动）
v[0] = 10;               // 随机访问 O(1)
v.reserve(1000);         // 预留容量，避免重复扩容
v.size();                // 当前元素数量
v.empty();               // 是否为空
v.clear();               // 清空
```

**unordered_map（键值查找）**：
```cpp
std::unordered_map<std::string, int> m;
m["apple"] = 5;          // 插入/修改
int val = m["apple"];    // 访问（不存在会创建默认值）
m.erase("apple");        // 删除
if (m.count("key")) {}   // 检查是否存在（返回 0 或 1）
// C++20: if (m.contains("key")) {}
```

**unordered_set（去重）**：
```cpp
std::unordered_set<int> s = {1, 2, 3, 2, 1};  // 自动去重
s.insert(4);             // 插入
s.erase(2);              // 删除
if (s.count(3)) {}       // 检查是否存在
```

**通用操作**：
```cpp
// 遍历（适用所有容器）
for (const auto& item : container) { /* ... */ }

// 大小
container.size();
container.empty();
container.clear();
```

**时间复杂度**：
| 容器 | 查找 | 插入 | 删除 |
|-----|------|------|------|
| vector | O(n) | O(1)尾部 | O(1)尾部 |
| unordered_map | O(1) | O(1) | O(1) |
| unordered_set | O(1) | O(1) | O(1) |
| map（有序） | O(log n) | O(log n) | O(log n) |

---

## 04. 移动语义

**核心概念**：移动 = 转移所有权，不拷贝数据（O(1)）

**左值 vs 右值**：
```cpp
int x = 10;
//  ↑   ↑
// 左值 右值

// 左值：有名字，可以取地址
int a = 5;  int* p = &a;  // ✅

// 右值：临时对象，不能取地址
int b = 10 + 20;  // 10 + 20 是右值
// int* p = &(10 + 20);  // ❌

// 右值马上销毁 → 可以"偷"走资源（移动）
```

**六个特殊成员函数**：
```cpp
Widget w1;           // 1. 默认构造
Widget w2(w1);       // 2. 拷贝构造（创建新对象，from 左值）
w3 = w1;             // 3. 拷贝赋值（已存在对象，from 左值）
Widget w4(move(w1)); // 4. 移动构造（创建新对象，from 右值）
w4 = move(w2);       // 5. 移动赋值（已存在对象，from 右值）
                     // 6. 析构
```

**移动构造/移动赋值实现**：
```cpp
class MyVector {
    int* data_;
    size_t size_;
public:
    // 移动构造
    MyVector(MyVector&& o) noexcept
        : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;  // "偷"走资源，掏空原对象
        o.size_ = 0;
    }

    // 移动赋值
    MyVector& operator=(MyVector&& o) noexcept {
        if (this != &o) {
            delete[] data_;       // 释放旧资源
            data_ = o.data_;      // 偷走新资源
            size_ = o.size_;
            o.data_ = nullptr;    // 掏空原对象
            o.size_ = 0;
        }
        return *this;
    }
};
```

**std::move**：
```cpp
// std::move 不移动，只是类型转换（左值 → 右值引用）
std::string s1 = "hello";
std::string s2 = std::move(s1);  // 强制移动，s1 被掏空

// ⚠️ 移动后不要再用原对象
// std::cout << s1;  // 危险
```

**何时用 std::move**：
```cpp
// ✅ 转移所有权
std::string s2 = std::move(s1);

// ✅ 容器中移动元素
vec.push_back(std::move(s));

// ❌ 返回局部变量时不要用（妨碍 RVO）
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return std::move(vec);  // ❌ 错误：破坏 RVO
}

// ✅ 正确：让编译器自动优化
std::vector<int> foo() {
    std::vector<int> vec(1000);
    return vec;  // 编译器自动优化（RVO 或移动）
}
```

**RVO（Return Value Optimization）**：
```
编译器的自动优化：直接在目标位置构造对象，零拷贝零移动

性能排序：
RVO/NRVO（编译器优化） > 移动 > 拷贝
   O(0)               O(1)   O(n)
   零开销              偷指针   复制数据
```

**引用类型**：
```cpp
T&                // 左值引用（绑定有名对象）
const T&          // 常量引用（函数参数首选）
T&&               // 右值引用（移动语义，绑定临时对象）
```

**关键要点**：
- 移动 = 转移所有权 O(1)，拷贝 = 复制数据 O(n)
- 返回局部变量自动移动/RVO，**不要写 std::move**
- 移动后的对象不要再用
- 移动函数必须标记 `noexcept`（否则 vector 扩容不用移动）
- const 对象不能移动（会退化为拷贝）

**常见陷阱**：
```cpp
// ❌ 使用被移动的对象
std::string s2 = std::move(s1);
std::cout << s1;  // 危险

// ❌ const 对象不能移动
const std::string s = "hello";
auto s2 = std::move(s);  // 实际是拷贝

// ❌ 返回时用 std::move
return std::move(vec);  // 妨碍 RVO

// ❌ 移动构造缺少 noexcept
MyClass(MyClass&& o) { }  // vector 扩容时不会用移动
```

---

## 05. Lambda

**基本语法**：
```cpp
[捕获](参数) { 函数体 }
```

**捕获列表**：
```cpp
int x = 10, y = 20;

[]          // 不捕获
[x]         // 按值捕获 x（拷贝，默认 const）
[&x]        // 按引用捕获 x（可修改外部变量）
[=]         // 按值捕获所有（安全）
[&]         // 按引用捕获所有
[=, &x]     // 默认按值，x 按引用
[&, x]      // 默认按引用，x 按值

// 示例
auto f1 = [x]() { return x + 1; };     // x 是拷贝
auto f2 = [&x]() { x = 100; };         // x 是引用，修改外部 x
auto f3 = [=]() { return x + y; };     // 捕获所有变量
```

**常用场景**：
```cpp
std::vector<int> v = {3, 1, 4, 1, 5};

// 1. 排序（降序）
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// 2. 查找
auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 3; });

// 3. 计数
int count = std::count_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });

// 4. 遍历
std::for_each(v.begin(), v.end(), [](int x) { std::cout << x << " "; });

// 5. 带捕获的使用
int threshold = 5;
auto count2 = std::count_if(v.begin(), v.end(),
                            [threshold](int x) { return x > threshold; });
```

**mutable 关键字**：
```cpp
int x = 10;
// 默认：按值捕获是 const，不能修改
// auto f = [x]() { x = 20; };  // 编译错误

auto f1 = [x]() mutable { x = 20; };  // mutable：修改拷贝（不影响外部）
f1();  // x 还是 10

auto f2 = [&x]() { x = 20; };  // 引用捕获：修改外部
f2();  // x 变成 20
```

**常见陷阱**：
```cpp
// ❌ 悬空引用：捕获引用后延迟调用
auto make_lambda() {
    int x = 10;
    return [&x]() { return x; };  // 危险：x 已销毁
}

// ✅ 按值捕获（安全）
auto make_lambda() {
    int x = 10;
    return [x]() { return x; };  // 安全：拷贝了 x
}
```

**要点**：
- 默认用 `[=]` 按值捕获（安全）
- 需要修改外部变量用 `[&]`
- 立即使用的小函数用 lambda
- 不要捕获引用后延迟调用（悬空引用）
- `mutable` 只修改 lambda 内部的拷贝

---

## 06. 模板

**核心思想**：一次编写，处处复用。编译器为每种类型生成代码。

**函数模板（自动推导类型）**：
```cpp
template<typename T>
T max(T a, T b) { return a > b ? a : b; }

max(3, 5);        // 自动推导 T = int
max(1.5, 2.5);    // 自动推导 T = double
max<int>(3, 5);   // 显式指定类型
```

**类模板（必须显式指定类型）**：
```cpp
template<typename T>
class Box {
    T value_;
public:
    Box(T v) : value_(v) {}
    T get() const { return value_; }
};

Box<int> b1(42);           // 必须指定类型
Box<std::string> b2("hi"); // 不能自动推导
```

**多个模板参数**：
```cpp
template<typename T, typename U>
auto add(T a, U b) { return a + b; }

add(3, 1.5);  // T=int, U=double, 返回 double
```

**变长模板（可变参数）**：
```cpp
template<typename... Args>     // Args 是参数包
void print(Args... args) {
    (std::cout << ... << args) << "\n";  // C++17 折叠表达式
}

print(1, 2, 3, "hello");  // 接受任意数量参数
```

**常见陷阱**：
```cpp
// ❌ 模板定义在 .cpp，使用在另一个文件
// foo.cpp
template<typename T>
T add(T a, T b) { return a + b; }
// main.cpp
add(3, 5);  // 链接错误！

// ✅ 模板定义必须在头文件（或同一文件）
// foo.h
template<typename T>
T add(T a, T b) { return a + b; }

// ❌ 类型推导失败
template<typename T>
T max(T a, T b) { return a > b ? a : b; }
max(3, 1.5);  // 错误：T 既是 int 又是 double

// ✅ 显式指定或用多个类型参数
max<double>(3, 1.5);
```

**要点**：
- 函数模板自动推导，类模板必须显式指定
- 模板定义必须在头文件（多文件项目）
- 标准库大量使用模板（`vector<T>`, `map<K,V>` 等）
- 简单场景用模板，复杂场景考虑其他方案

---

> 阶段 2：并发编程 🔄

## 07. 线程基础

**核心概念**：线程 = 独立的执行流，共享进程内存

**创建线程**：
```cpp
std::thread t(函数);          // 函数
std::thread t([]{...});       // Lambda（推荐）
std::thread t(Worker{});      // 函数对象
```

**join vs detach**：
```cpp
t.join();    // 等待线程结束（推荐）
t.detach();  // 分离线程（慎用，易悬空引用）
```

**传递参数**：
```cpp
std::thread t(func, arg1, arg2);      // 按值
std::thread t(func, std::ref(var));   // 按引用（必须用 std::ref）
std::thread t([x]{...});              // Lambda 捕获
```

**线程信息**：
```cpp
std::thread::hardware_concurrency();  // CPU 核心数
std::this_thread::get_id();           // 当前线程 ID
std::this_thread::sleep_for(std::chrono::seconds(1));  // 休眠
```

**RAII 线程管理**：
```cpp
class ThreadGuard {
    std::thread& t_;
public:
    explicit ThreadGuard(std::thread& t) : t_(t) {}
    ~ThreadGuard() { if (t_.joinable()) t_.join(); }
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// C++20
std::jthread t([]{...});  // 析构时自动 join
```

**常见陷阱**：
```cpp
// ❌ 忘记 join/detach
std::thread t([]{...});
// 离开作用域 → std::terminate，程序崩溃

// ❌ 引用捕获 + detach
int x = 10;
std::thread t([&x]{...});
t.detach();  // x 销毁，悬空引用

// ❌ 重复 join
t.join();
t.join();  // 崩溃
```

**要点**：
- 线程创建后必须 join 或 detach
- detach 时按值捕获局部变量
- 线程数 ≈ CPU 核心数（过多性能下降）
- 用 RAII 管理线程（避免忘记 join）

---

## 08. 互斥锁

**核心问题**：多线程同时修改共享数据 → 数据竞争

**基本用法**：
```cpp
std::mutex mtx;
int counter = 0;

mtx.lock();
counter++;
mtx.unlock();
```

**三种 RAII 锁**：
```cpp
// 1. lock_guard（推荐，90%情况）
{
    std::lock_guard<std::mutex> lock(mtx);  // 构造时加锁
    counter++;
}  // 析构时自动解锁

// 2. unique_lock（灵活，可手动控制）
std::unique_lock<std::mutex> lock(mtx);
lock.unlock();  // 手动解锁
// ... 不需要锁的操作 ...
lock.lock();    // 再次加锁

// 3. scoped_lock（C++17，多个锁）
std::scoped_lock lock(mtx1, mtx2);  // 同时锁定，避免死锁
```

**锁的选择**：
- 简单场景 → `lock_guard`
- 需要手动控制或配合条件变量 → `unique_lock`
- 多个锁 → `scoped_lock`（C++17）

**死锁问题**：
```cpp
// ❌ 死锁
void thread1() {
    std::lock_guard<std::mutex> lock1(mtx1);  // 持有 mtx1
    std::lock_guard<std::mutex> lock2(mtx2);  // 等待 mtx2
}
void thread2() {
    std::lock_guard<std::mutex> lock2(mtx2);  // 持有 mtx2
    std::lock_guard<std::mutex> lock1(mtx1);  // 等待 mtx1
}
// 互相等待，永远阻塞

// ✅ 解决：固定加锁顺序
void both_threads() {
    std::lock_guard<std::mutex> lock1(mtx1);  // 都先 mtx1
    std::lock_guard<std::mutex> lock2(mtx2);  // 都后 mtx2
}

// ✅ 解决：用 scoped_lock
std::scoped_lock lock(mtx1, mtx2);  // 自动避免死锁
```

**性能建议**：
```cpp
// ✅ 好：锁的范围小
{
    std::lock_guard<std::mutex> lock(mtx);
    data.push_back(value);  // 只锁关键操作
}
expensive_computation();  // 不需要锁

// ❌ 坏：锁的范围大
{
    std::lock_guard<std::mutex> lock(mtx);
    data.push_back(value);
    expensive_computation();  // 浪费，其他线程等待
}
```

**常见陷阱**：
```cpp
// ❌ 忘记加锁
counter++;  // 数据竞争

// ❌ 手动 lock/unlock（易忘记）
mtx.lock();
if (error) return;  // 忘记 unlock，死锁
mtx.unlock();

// ❌ 返回被保护数据的引用
std::vector<int>& get_data() {
    std::lock_guard<std::mutex> lock(mtx);
    return vec;  // 锁解除，但引用还在外面用
}
```

**要点**：
- 多线程访问共享数据必须加锁
- 优先用 `lock_guard`（90%情况）
- 锁的范围尽量小（性能）
- 固定加锁顺序（避免死锁）
- 用 RAII 管理锁（永远不要手动 lock/unlock）
- 多个锁用 `scoped_lock`（C++17）

---

## 09. 条件变量

**核心问题**：线程如何高效等待条件？（不用忙等）

**基本用法**：
```cpp
std::mutex mtx;
std::condition_variable cv;
bool ready = false;

// 等待线程
std::unique_lock<std::mutex> lock(mtx);
cv.wait(lock, []{ return ready; });  // 阻塞，直到 ready 为 true

// 通知线程
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}
cv.notify_one();  // 唤醒一个等待的线程
```

**wait 的行为**：
1. 检查条件，为真立即返回
2. 为假 → 解锁 mutex，线程休眠（不占 CPU）
3. 被唤醒 → 重新加锁，再次检查条件
4. 条件为真才返回

**为什么必须用 unique_lock？**
```cpp
// ❌ 不能用 lock_guard
std::lock_guard<std::mutex> lock(mtx);
cv.wait(lock);  // 编译错误

// ✅ 必须用 unique_lock
std::unique_lock<std::mutex> lock(mtx);
cv.wait(lock);  // wait 需要临时解锁
```

**wait 的三种形式**：
```cpp
// 1. 带谓词（推荐，自动处理虚假唤醒）
cv.wait(lock, []{ return ready; });

// 2. 不带谓词（需要手动循环）
while (!ready) {
    cv.wait(lock);
}

// 3. 带超时
bool result = cv.wait_for(lock, std::chrono::seconds(1), []{ return ready; });
if (result) {
    // 条件满足
} else {
    // 超时
}
```

**notify_one vs notify_all**：
```cpp
cv.notify_one();   // 唤醒一个线程（单消费者）
cv.notify_all();   // 唤醒所有线程（多消费者）
```

**生产者-消费者模型**：
```cpp
std::queue<int> buffer;
std::mutex mtx;
std::condition_variable cv;
const int MAX_SIZE = 10;

// 生产者
void producer() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return buffer.size() < MAX_SIZE; });  // 等待不满
    buffer.push(data);
    lock.unlock();
    cv.notify_all();  // 通知消费者
}

// 消费者
void consumer() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return !buffer.empty(); });  // 等待不空
    int data = buffer.front();
    buffer.pop();
    lock.unlock();
    cv.notify_all();  // 通知生产者
}
```

**虚假唤醒**：
```cpp
// ❌ 危险：不检查条件
cv.wait(lock);
// 可能虚假唤醒，ready 不一定为 true

// ✅ 安全：总是用谓词
cv.wait(lock, []{ return ready; });
```

**通知时机**：
```cpp
// ✅ 好：先解锁再通知（性能更好）
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}  // 解锁
cv.notify_one();

// ⚠️ 可以但不推荐：持有锁时通知
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
    cv.notify_one();  // 等待线程被唤醒，但立即被锁阻塞
}
```

**常见陷阱**：
```cpp
// ❌ 忘记检查条件（虚假唤醒）
cv.wait(lock);
int value = buffer.front();  // buffer 可能为空

// ❌ 用 lock_guard
std::lock_guard<std::mutex> lock(mtx);
cv.wait(lock);  // 编译错误

// ❌ 修改条件后不通知
{
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
}
// 忘记 cv.notify_one()，等待线程永远阻塞
```

**要点**：
- 条件变量用于线程间等待/通知
- 必须配合 `unique_lock` 使用
- 总是用谓词检查条件（避免虚假唤醒）
- 修改条件后立即通知
- 先解锁再通知（性能更好）
- 单消费者用 `notify_one`，多消费者用 `notify_all`

---

## 10. 原子操作

**核心概念**：原子操作 = 不可分割的操作，无需锁，硬件直接支持

**为什么需要？**
```cpp
// ❌ 非原子：counter++ 分三步（读取 → 加1 → 写回）
int counter = 0;
counter++;  // 多线程不安全

// ✅ 原子：一步完成，线程安全
std::atomic<int> counter(0);
counter++;  // 快！无需锁
```

**基本用法**：
```cpp
#include <atomic>

std::atomic<int> a(0);
std::atomic<bool> flag(false);
std::atomic<int*> ptr(nullptr);

// 读取
int value = a.load();
int value2 = a;  // 隐式 load()

// 写入
a.store(10);
a = 10;  // 隐式 store()
```

**常用操作**：
```cpp
std::atomic<int> counter(0);

counter++;              // 自增
counter--;              // 自减
counter += 5;           // 加 5

int old = counter.fetch_add(1);    // 返回旧值，然后 +1
int old2 = counter.exchange(100);  // 设为 100，返回旧值

// CAS：compare_exchange（最强大）
int expected = 10;
int desired = 20;
bool success = counter.compare_exchange_strong(expected, desired);
// 如果 counter == expected，设为 desired，返回 true
// 否则，expected 被更新为 counter 的当前值，返回 false
```

**原子 bool（标志位）**：
```cpp
std::atomic<bool> ready(false);

ready.store(true);              // 设置
bool value = ready.load();      // 读取
bool old = ready.exchange(true); // 交换
```

**自旋锁实现**：
```cpp
class SpinLock {
    std::atomic<bool> flag_{false};
public:
    void lock() {
        while (flag_.exchange(true)) {
            // 自旋等待
        }
    }
    void unlock() {
        flag_.store(false);
    }
};
```

**CAS（比较并交换）**：
```cpp
// 无锁队列的核心
void push(int value) {
    Node* new_node = new Node(value);
    new_node->next = head.load();

    // CAS 循环：不断重试，直到成功
    while (!head.compare_exchange_weak(new_node->next, new_node)) {
        // 失败：其他线程抢先了，重试
    }
}
```

**weak vs strong**：
```cpp
compare_exchange_weak    // 可能虚假失败，用于循环（性能更好）
compare_exchange_strong  // 不会虚假失败，用于单次操作
```

**内存顺序**：
```cpp
memory_order_seq_cst    // 默认：最强，最安全（推荐初学者）
memory_order_acquire    // 读操作
memory_order_release    // 写操作
memory_order_relaxed    // 最弱：只保证原子性，性能最好

// 生产者-消费者
int data = 0;
std::atomic<bool> ready(false);

// 生产者
data = 42;
ready.store(true, std::memory_order_release);  // 写操作

// 消费者
while (!ready.load(std::memory_order_acquire)) {}  // 读操作
std::cout << data;  // 保证看到 42
```

**原子操作 vs 锁**：
```cpp
// ✅ 原子操作（适用场景）
std::atomic<int> counter(0);      // 简单计数器
std::atomic<bool> done(false);    // 标志位
int old = value.exchange(10);     // 简单读-改-写

// ✅ 锁（适用场景）
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);
    data1 = 10;  // 保护多个变量
    data2 = 20;
    data3 = 30;
}
```

**选择建议**：
| 场景 | 使用 |
|------|------|
| 简单计数器/标志位 | `atomic` |
| 保护多个变量 | `mutex + lock_guard` |
| 复杂操作 | `mutex + lock_guard` |
| 性能关键的简单操作 | `atomic` |

**性能对比**：
- 原子操作：10-100 倍快于锁（简单操作）
- 原因：CPU 硬件直接支持，不进入内核

**常见陷阱**：
```cpp
// ❌ 非原子的复合操作
std::atomic<int> counter(0);
if (counter == 0) {
    counter = 1;  // 其他线程可能在这之间修改
}

// ✅ 用 CAS
int expected = 0;
counter.compare_exchange_strong(expected, 1);

// ❌ 以为能保护其他变量
std::atomic<bool> ready(false);
int data = 0;  // 不是原子的
data = 42;     // 数据竞争
ready = true;

// ✅ 用内存顺序或锁
data = 42;
ready.store(true, std::memory_order_release);

// ❌ 复杂类型
struct MyStruct { int a, b, c; };
std::atomic<MyStruct> s;  // 可能不支持

// ✅ 简单类型
std::atomic<int> a(0);
std::atomic<bool> b(false);
std::atomic<int*> ptr(nullptr);
```

**要点**：
- 原子操作 = 无锁同步，比锁快
- 用于简单类型（int、bool、指针）
- 常用：`load`、`store`、`fetch_add`、`exchange`、`compare_exchange`
- CAS 是最强大的原子操作，用于无锁数据结构
- 简单操作用原子，复杂操作用锁
- 内存顺序：初学者用默认，性能关键再优化

---

## 11. 异步编程

**核心概念**：异步 = 不等待结果，继续做其他事

**std::async - 启动异步任务（最简单）**：
```cpp
#include <future>

// 启动异步任务，立即返回
std::future<int> fut = std::async([]{
    return 42;
});

// 主线程继续做其他事...

// 需要时获取结果（阻塞）
int result = fut.get();
```

**启动策略**：
```cpp
// 1. async：立即创建新线程
auto fut1 = std::async(std::launch::async, task);

// 2. deferred：延迟执行（调用 get 时才执行）
auto fut2 = std::async(std::launch::deferred, task);

// 3. 默认：由实现决定
auto fut3 = std::async(task);
```

**std::future - 获取结果**：
```cpp
std::future<int> fut = std::async([]{ return 42; });

// 获取结果（只能调用一次）
int result = fut.get();  // 阻塞，直到完成
// fut.get();  // ❌ 错误：不能重复调用

// 等待（不获取结果）
fut.wait();  // 阻塞

// 等待一段时间
auto status = fut.wait_for(std::chrono::seconds(1));
if (status == std::future_status::ready) {
    // 任务完成
} else if (status == std::future_status::timeout) {
    // 超时
}
```

**std::promise - 手动设置结果**：
```cpp
std::promise<int> prom;
std::future<int> fut = prom.get_future();

// 生产者线程
std::thread t([&prom]{
    prom.set_value(42);  // 设置结果
});

// 消费者线程
int result = fut.get();  // 阻塞，直到 promise 设置值

t.join();
```

**promise/future 关系**：
- `promise` = 生产者（设置结果）
- `future` = 消费者（获取结果）
- 它们是一对

**设置异常**：
```cpp
std::promise<int> prom;
std::future<int> fut = prom.get_future();

try {
    throw std::runtime_error("错误");
} catch (...) {
    prom.set_exception(std::current_exception());
}

try {
    fut.get();  // 抛出异常
} catch (const std::exception& e) {
    std::cout << e.what();
}
```

**std::packaged_task - 包装函数**：
```cpp
// 包装函数
std::packaged_task<int(int, int)> task([](int a, int b) {
    return a + b;
});

// 获取 future
std::future<int> fut = task.get_future();

// 在线程中执行
std::thread t(std::move(task), 10, 20);

// 获取结果
int result = fut.get();  // 30

t.join();
```

**std::shared_future - 多个消费者**：
```cpp
std::future<int> fut = std::async([]{ return 42; });
std::shared_future<int> sf = fut.share();  // 转换

// 多个线程都可以获取结果
std::thread t1([sf]{ std::cout << sf.get(); });
std::thread t2([sf]{ std::cout << sf.get(); });
std::thread t3([sf]{ std::cout << sf.get(); });

t1.join();
t2.join();
t3.join();
```

**并行计算**：
```cpp
std::vector<std::future<int>> futures;

for (int i = 0; i < 10; ++i) {
    futures.push_back(std::async(std::launch::async, [i]{
        return compute(i);
    }));
}

// 收集结果
for (auto& fut : futures) {
    int result = fut.get();
}
```

**async vs thread**：
```cpp
// ✅ async（推荐，简洁）
auto fut = std::async([]{ return 42; });
int result = fut.get();

// ❌ thread（复杂）
int result;
std::thread t([&result]{ result = 42; });
t.join();
```

**选择**：
| 场景 | 使用 |
|------|------|
| 简单异步任务 | `async` |
| 需要精确控制线程 | `thread` |
| 手动控制结果 | `promise` |
| 线程池 | `packaged_task` |
| 多个消费者 | `shared_future` |

**常见陷阱**：
```cpp
// ❌ future 析构会阻塞
{
    auto fut = std::async(std::launch::async, long_task);
}  // fut 析构，阻塞等待

// ❌ 重复 get
std::future<int> fut = std::async([]{ return 42; });
int r1 = fut.get();
// int r2 = fut.get();  // 崩溃

// ✅ 用 shared_future
std::shared_future<int> sf = std::async([]{ return 42; }).share();
int r1 = sf.get();
int r2 = sf.get();  // 正确

// ❌ promise 忘记设置值
std::promise<int> prom;
std::future<int> fut = prom.get_future();
// prom 析构，fut.get() 会抛异常

// ✅ 确保设置值
prom.set_value(42);
```

**要点**：
- `async` - 启动异步任务（最简单）
- `future` - 获取结果（get 只能调用一次）
- `promise` - 手动设置结果（配合 future）
- `packaged_task` - 包装函数（用于线程池）
- `shared_future` - 多个消费者（可以多次 get）
- 简单异步用 `async`，精确控制用 `thread`
- future 析构会阻塞，记得调用 get