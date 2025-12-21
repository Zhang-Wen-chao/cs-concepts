# 性能优化

> 识别瓶颈、测量性能、应用优化技术

## 核心原则

**优化三步骤**：
1. **测量**：找到瓶颈（不要猜）
2. **优化**：针对瓶颈优化
3. **再测量**：验证效果

**过早优化是万恶之源** - Donald Knuth

## 性能测量

### 简单计时

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// 要测量的代码
do_work();

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "耗时: " << duration.count() << "ms\n";
```

### RAII 计时器

```cpp
class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    const char* name_;
public:
    Timer(const char* name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << ": " << duration.count() << " μs\n";
    }
};

// 使用
{
    Timer t("排序");
    std::sort(vec.begin(), vec.end());
}  // 自动打印耗时
```

## CPU 缓存优化

### 连续内存访问

```cpp
// ❌ 慢：跳跃访问，缓存未命中
struct Node {
    int data;
    Node* next;  // 指针指向随机位置
};

// 访问链表：缓存未命中率高
for (Node* p = head; p; p = p->next) {
    sum += p->data;
}

// ✅ 快：连续访问，缓存友好
std::vector<int> vec(10000);
for (int x : vec) {
    sum += x;  // 连续内存，缓存命中率高
}
```

**性能差异**：vector 比链表快 10-100 倍

### 数据结构布局

```cpp
// ❌ AoS (Array of Structures)：缓存浪费
struct Particle {
    float x, y, z;       // 位置
    float vx, vy, vz;    // 速度
    float r, g, b;       // 颜色
};
std::vector<Particle> particles;

// 只用位置时，速度和颜色也被加载到缓存（浪费）
for (auto& p : particles) {
    p.x += 1.0f;
}

// ✅ SoA (Structure of Arrays)：缓存高效
struct Particles {
    std::vector<float> x, y, z;       // 位置
    std::vector<float> vx, vy, vz;    // 速度
    std::vector<float> r, g, b;       // 颜色
};

// 只加载需要的数据
for (size_t i = 0; i < particles.x.size(); ++i) {
    particles.x[i] += 1.0f;
}
```

## 减少拷贝

### 移动语义

```cpp
// ❌ 拷贝
std::vector<int> get_data() {
    std::vector<int> vec(1000000);
    return vec;  // 现代 C++ 会自动移动，不拷贝
}

// ✅ 返回值优化（RVO）
// 编译器自动优化，无需 std::move
auto data = get_data();  // 快！
```

### 引用传递

```cpp
// ❌ 拷贝大对象
void process(std::vector<int> vec) {  // 拷贝整个 vector
    // ...
}

// ✅ 引用传递
void process(const std::vector<int>& vec) {  // 不拷贝
    // ...
}
```

### emplace vs push

```cpp
std::vector<std::string> vec;

// ❌ 慢：构造临时对象 → 移动
vec.push_back(std::string("hello"));

// ✅ 快：原地构造
vec.emplace_back("hello");
```

## 避免动态分配

### 小字符串优化

```cpp
// std::string 小字符串不分配堆内存
std::string s1 = "short";     // 栈上（快）
std::string s2 = "very long string...";  // 堆上（慢）
```

### 栈上数组

```cpp
// ❌ 堆分配
std::vector<int> vec(10);

// ✅ 栈上（小数组）
std::array<int, 10> arr;  // 编译期大小，栈上
```

### reserve 预分配

```cpp
std::vector<int> vec;

// ❌ 多次扩容
for (int i = 0; i < 10000; ++i) {
    vec.push_back(i);  // 可能触发扩容
}

// ✅ 预分配
vec.reserve(10000);  // 一次分配
for (int i = 0; i < 10000; ++i) {
    vec.push_back(i);  // 不扩容
}
```

## 循环优化

### 提升不变量

```cpp
// ❌ 每次循环都计算
for (int i = 0; i < vec.size(); ++i) {
    result += vec[i] * expensive_function();  // 重复计算
}

// ✅ 提升到循环外
int factor = expensive_function();
for (int i = 0; i < vec.size(); ++i) {
    result += vec[i] * factor;
}
```

### 缓存 size

```cpp
// ❌ 每次调用 size()
for (size_t i = 0; i < vec.size(); ++i) {  // size() 很快，但不必要
    // ...
}

// ✅ 缓存（现代编译器会优化）
size_t n = vec.size();
for (size_t i = 0; i < n; ++i) {
    // ...
}

// ✅✅ 最佳：范围循环
for (auto& item : vec) {
    // ...
}
```

### 循环展开

```cpp
// ❌ 普通循环
for (int i = 0; i < n; ++i) {
    result += data[i];
}

// ✅ 手动展开（减少分支预测）
for (int i = 0; i < n; i += 4) {
    result += data[i];
    result += data[i + 1];
    result += data[i + 2];
    result += data[i + 3];
}
// 现代编译器会自动展开
```

## 算法优化

### 选择合适的数据结构

```cpp
// 查找频繁
// ❌ vector：O(n)
std::vector<int> vec;
std::find(vec.begin(), vec.end(), target);

// ✅ unordered_set：O(1)
std::unordered_set<int> set;
set.count(target);
```

### 减少不必要的操作

```cpp
// ❌ 每次都排序
for (int i = 0; i < 100; ++i) {
    vec.push_back(data[i]);
    std::sort(vec.begin(), vec.end());  // 100 次排序
}

// ✅ 最后排序一次
for (int i = 0; i < 100; ++i) {
    vec.push_back(data[i]);
}
std::sort(vec.begin(), vec.end());  // 1 次
```

## 并发优化

### 多线程

```cpp
// 串行处理
for (auto& item : items) {
    process(item);
}

// 并行处理
std::vector<std::thread> threads;
size_t chunk = items.size() / num_threads;
for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i] {
        for (size_t j = i * chunk; j < (i + 1) * chunk; ++j) {
            process(items[j]);
        }
    });
}
for (auto& t : threads) {
    t.join();
}
```

### 原子操作代替锁

```cpp
// ❌ 锁（慢）
std::mutex mtx;
int counter = 0;
mtx.lock();
counter++;
mtx.unlock();

// ✅ 原子操作（快 10-100 倍）
std::atomic<int> counter(0);
counter++;
```

## 编译器优化

### 优化级别

```bash
# 无优化（调试）
g++ -O0 code.cpp

# 基本优化
g++ -O1 code.cpp

# 推荐优化
g++ -O2 code.cpp

# 激进优化
g++ -O3 code.cpp

# 针对本机 CPU
g++ -O3 -march=native code.cpp
```

### 内联

```cpp
// 编译器会自动内联小函数
inline int square(int x) {
    return x * x;
}

// 调用
int y = square(5);  // 编译器直接替换为 25
```

## I/O 优化

### 批量读写

```cpp
// ❌ 逐字节读
for (int i = 0; i < 1000000; ++i) {
    char c;
    file.read(&c, 1);  // 100 万次系统调用
}

// ✅ 批量读
std::vector<char> buffer(1000000);
file.read(buffer.data(), buffer.size());  // 1 次系统调用
```

### 关闭同步

```cpp
// C++ 流与 C 的 stdio 同步（慢）
std::ios::sync_with_stdio(false);  // 关闭同步，加速
std::cin.tie(nullptr);  // 解除 cin 和 cout 的绑定
```

## 性能分析工具

**Linux**：
```bash
# perf：CPU 性能分析
perf record ./program
perf report

# valgrind：内存分析
valgrind --tool=callgrind ./program
```

**macOS**：
```bash
# Instruments：Xcode 自带
instruments -t "Time Profiler" ./program
```

## 常见陷阱

### 陷阱 1：过早优化

```cpp
// ❌ 不知道瓶颈在哪就优化
void process() {
    // 优化了这里，但瓶颈在别处
}

// ✅ 先测量，找到瓶颈
// perf 显示瓶颈在排序
std::sort(...);  // 针对性优化
```

### 陷阱 2：忽略编译器优化

```cpp
// ❌ 手动优化，反而妨碍编译器
int sum = 0;
for (int i = 0; i < n; ++i) {
    sum += data[i];
}
// 编译器可能向量化（SIMD）

// ❌ 过度优化
// 复杂的手动展开，编译器已经做了
```

### 陷阱 3：微观优化，忽略算法

```cpp
// ❌ 优化循环，但算法是 O(n²)
for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
        // 即使优化，还是 O(n²)
    }
}

// ✅ 改算法为 O(n log n)
std::sort(...);  // 更大的提升
```

## 优化优先级

**1. 算法（最重要）**：
- O(n²) → O(n log n)：提升 1000 倍

**2. 数据结构**：
- 链表 → vector：提升 10-100 倍

**3. 内存管理**：
- 频繁 new/delete → 对象池：提升 10-100 倍

**4. 并发**：
- 单线程 → 多线程：提升 2-8 倍

**5. 微观优化**：
- 循环展开、内联：提升 10-30%

## 核心要点

1. **先测量，再优化**：
   - 不要猜瓶颈
   - 用工具测量

2. **算法最重要**：
   - O(n²) → O(n log n) 比微优化有效得多

3. **缓存友好**：
   - 连续内存访问（vector）
   - 避免指针跳跃（链表）

4. **减少拷贝**：
   - 引用传递
   - 移动语义
   - emplace

5. **减少分配**：
   - reserve 预分配
   - 对象池/内存池
   - 栈上数组

6. **并发**：
   - CPU 密集型用多线程
   - 简单计数用原子操作

7. **编译器优化**：
   - -O2 或 -O3
   - 现代编译器很聪明

8. **优化优先级**：
   算法 > 数据结构 > 内存管理 > 并发 > 微观优化
