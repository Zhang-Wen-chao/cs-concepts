# Stage 2: 内存模型与 RAII

> 这是 C++ **跟 Python 拉开最大差距**的一节。  
> 核心问题：**内存谁负责释放？怎么保证一定释放？**

## 大图：内存从哪来、到哪去

```
┌─────────────── 程序的内存 ────────────────┐
│                                            │
│  ┌──────────────┐  ┌────────────────────┐  │
│  │   代码段      │  │    数据段 / BSS    │  │
│  │  (text)      │  │  (data/bss)        │  │
│  │ 编译后指令    │  │ 全局变量、静态变量  │  │
│  └──────────────┘  └────────────────────┘  │
│                                            │
│  ┌──────────────┐  ┌────────────────────┐  │
│  │     栈       │  │       堆           │  │
│  │   (stack)    │  │     (heap)         │  │
│  │ 局部变量      │  │ new/malloc 出来的  │  │
│  │ 自动分配释放  │  │ 手动分配，手动释放  │  │
│  │ LIFO（Last-In-First-Out / 后进先出）顺序 │  │ 任意顺序 │  │
│  │ 几 MB 大小   │  │ 几 GB 大小         │  │
│  └──────────────┘  └────────────────────┘  │
│                                            │
└────────────────────────────────────────────┘
```

## 栈 (Stack) vs 堆 (Heap)

|  | 栈 (Stack) | 堆 (Heap) |
|---|---|---|
| **分配方式** | 进入作用域自动 | `new` / `malloc` 手动 |
| **释放方式** | 离开作用域自动 | `delete` / `free` 手动 |
| **大小** | 几 MB（默认 8MB 左右） | 几 GB（受系统内存限制） |
| **速度** | **极快**（移动栈指针） | 慢（要找空闲块） |
| **碎片** | 不会产生 | 容易产生 |
| **生命周期** | 严格 LIFO（后进先出） | 任意 |
| **线程** | 每个线程独立 | 全进程共享 |
| **Python 对应** | 局部变量（CPython 里也在栈） | 对象（Python 解释器代你管） |

## 栈长什么样

Stage 0 你已经看到了汇编里的栈操作：
```asm
sub sp, sp, #32    ; 栈指针 sp 减 32 = 分配 32 字节
stp x29, x30, [sp, #16]  ; 在栈上保存两个寄存器
```

**C++ 函数进入时**：
1. 栈指针下移（腾地方）
2. 把"调用者需要的寄存器"保存到栈上
3. 局部变量分配在栈上
4. 函数结束，栈指针上移（释放）→ **自动！**

**栈帧**长这样（main 函数栈帧）：
```
┌────────────────────┐ ← 栈底（高地址）
│ main 的局部变量      │
├────────────────────┤
│ main 保存的寄存器    │
├────────────────────┤
│ main 调用前的栈顶    │
│ ...                 │
```

## 栈 vs 堆：真实代码

```cpp
void stack_vs_heap() {
    int a = 42;                  // 栈上：自动分配/释放
    int arr[100];                // 栈上：100 * 4 = 400 字节

    int* p = new int(42);        // 堆上：手动分配
    int* arr_p = new int[100];   // 堆上：手动分配

    // ... 用 p 和 arr_p ...

    delete p;                    // 手动释放
    delete[] arr_p;              // 手动释放数组
}  // 函数结束，a、arr 自动释放；p 指向的内存已经手动释放
```

**问题来了**：
- `delete` 忘了 → 内存泄漏
- `delete` 后还用 p → 悬空指针 (use-after-free)
- `delete` 两次 → 双重释放 (double-free)，程序直接崩
- 异常抛出，绕过 `delete` → 泄漏

**这就是 C++ 的"难"——你**得**管这些。**

---

## 解法 1：RAII（Resource Acquisition Is Initialization）

**核心理念**：把"资源的获取"绑到**对象构造**，把"资源的释放"绑到**对象析构**。对象生命周期一结束，资源自动释放。

```cpp
// 传统写法：手动管理
void old_way() {
    FILE* f = fopen("data.txt", "r");
    if (!f) return;
    // ... 用 f 干很多事 ...
    if (error) {
        fclose(f);   // ❌ 每个 return 前都要关
        return;
    }
    fclose(f);       // ❌ 还有可能忘
}

// RAII 写法：自动管理
void modern_way() {
    std::ifstream f("data.txt");  // 构造：打开文件
    if (!f.is_open()) return;
    // ... 用 f 干很多事 ...
    // 离开作用域，自动调用 f 的析构函数，关闭文件
    // 哪怕中间抛异常，析构函数也保证被调用
}
```

**关键洞察**：C++ **保证**对象离开作用域时调用析构函数——即使有异常。所以**只要把资源绑到析构函数**，就**永远不漏**。

### 自己写一个 RAII 包装

```cpp
class FileHandle {
    FILE* file_;
public:
    // 构造：获取资源
    FileHandle(const char* name) : file_(std::fopen(name, "r")) {
        if (!file_) throw std::runtime_error("Cannot open file");
    }
    // 析构：释放资源
    ~FileHandle() {
        if (file_) std::fclose(file_);
    }
    // 禁用拷贝（防 double-free，见下）
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    FILE* get() { return file_; }
};
```

**使用**：
```cpp
void process_file() {
    FileHandle f("data.txt");  // 打开
    // ... 干很多事，可能抛异常 ...
}  // 自动 fclose，哪怕抛异常
```

## RAII 的威力：异常安全

```cpp
// ❌ 异常不安全：异常时 leak
void unsafe(int n) {
    int* p = new int[n];
    if (n < 0) throw std::bad_alloc();  // 异常！p 泄漏
    delete[] p;
}

// ✅ 异常安全：异常时不 leak
void safe(int n) {
    std::vector<int> v(n);  // vector 也是 RAII
    if (n < 0) throw std::bad_alloc();  // v 自动析构，释放
    // ...
}
```

**vector、智能指针、string、ifstream、lock_guard……全是 RAII**。C++ 标准库**到处都是 RAII**。

---

## 解法 2：智能指针 (Smart Pointers)

智能指针就是**帮你管堆内存的 RAII 包装**。C++11 起，**3 种**：

| 智能指针 | 所有权 | 拷贝 | 适用场景 |
|---|---|---|---|
| `unique_ptr<T>` | 独占 | **不能**拷贝（只能 move） | 90% 场景，**首选** |
| `shared_ptr<T>` | 共享 | 可以 | 需要共享所有权时 |
| `weak_ptr<T>` | 不持有 | — | 打破 `shared_ptr` 循环引用 |

### unique_ptr：独占所有权（90% 用这个）

```cpp
// ❌ 旧写法：裸指针
void old() {
    int* p = new int(42);
    // ... 用 p ...
    delete p;  // 容易忘
}

// ✅ 现代写法：unique_ptr
void modern() {
    std::unique_ptr<int> p = std::make_unique<int>(42);
    // ... 用 *p 取值，p->xxx 调用成员 ...
    // 离开作用域，自动 delete
}
```

**关键 API**：
- `std::make_unique<T>(args...)` — 创建（C++14 起，**首选**）
- `*p` / `p->xxx` — 跟裸指针一样用
- `p.get()` — 拿原始指针（**不推荐**，要保证不 delete）
- `p.reset(newPtr)` — 换管理的对象
- `std::move(p)` — 转移所有权（详见移动语义）

### shared_ptr：共享所有权

```cpp
std::shared_ptr<int> p1 = std::make_shared<int>(42);
std::shared_ptr<int> p2 = p1;  // 引用计数：2
p1.reset();                    // 引用计数：1
p2.reset();                    // 引用计数：0 → 自动 delete
```

**开销**：每次拷贝/释放都要原子操作维护引用计数（**比 unique_ptr 慢**）。**没有共享需求就不要用**。

**循环引用问题**（面试常考）：
```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev;
};
Node a, b;
a.next = b;       // b 引用计数：1
b.prev = a;       // a 引用计数：1
// 函数结束，a、b 局部变量析构 → 引用计数：1 → 永远不释放！
```

**解法**：用 `weak_ptr` 打破循环：
```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // weak 不增加引用计数
};
```

---

## 解法 3：移动语义 (Move Semantics)

**问题**：值语义拷贝开销大
```cpp
std::vector<int> v(1000000, 0);
std::vector<int> copy = v;  // 复制 100 万元素！慢！
```

**解法**：移动（**转移所有权，不复制数据**）：
```cpp
std::vector<int> v(1000000, 0);
std::vector<int> moved = std::move(v);  // O(1)：swap 内部指针
// v 现在是空的，moved 拥有原数据
```

**底层**：
- 拷贝：申请新内存，复制 100 万元素
- 移动：把 v 内部的指针交给 moved，v 指向空

**关键点**：
- `std::move` **不**移动任何东西，**只是把左值转成右值**（告诉编译器"这个对象可以被人接管"）
- 真正移动是**移动构造 / 移动赋值**干的事

**右值引用** `&&`：
```cpp
class Buffer {
    int* data_;
    size_t size_;
public:
    // 移动构造
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;  // 把 other 清空
        other.size_ = 0;
    }
};
```

**面试必问**：
- "拷贝构造 vs 移动构造" → 答：拷贝构造**深拷贝数据**（O(n)），移动构造**转移指针**（O(1)）
- "什么时候用 `std::move`" → 答：知道对象不再使用时（函数 return、临时对象）
- "移动构造为什么加 `noexcept`" → 答：标准库容器（vector）会优先用 noexcept 移动，否则退化成拷贝

---

## 和训练/推理的连接

- **Megatron / vLLM / TensorRT** 全部大量用 `unique_ptr` / `shared_ptr`
- **CUDA stream** 用 RAII 包装（`std::lock_guard` 同理）
- **TensorRT 的 IExecutionContext** 是 move-only 的，必须用 `std::move` 转移
- **看懂这些库的源码**，RAII + 智能指针 + move 是基础
- **自定义 CUDA kernel** 时，`cudaMalloc` / `cudaFree` 也建议自己包一层 RAII（不然 GPU 内存泄漏比 CPU 还难查）

---

## 实战任务

`practices/02_raii_smart_pointers.cpp`（我马上写）：
- 演示裸指针的 3 种典型 bug（泄漏、双重释放、悬空）
- 演示 `unique_ptr` 怎么解决
- 演示 `shared_ptr` + `weak_ptr` 解决循环引用
- 演示 `std::move` 跟拷贝的性能差
- 演示自己写 RAII 包装

## 自测问题

1. 栈和堆的区别？什么时候用哪个？
2. RAII 是什么？为什么 C++ 特别需要它？（Python 不用）
3. `unique_ptr` 跟 `shared_ptr` 怎么选？`weak_ptr` 干嘛用？
4. `std::move` 到底干了什么？它真的移动数据了吗？
5. 拷贝构造和移动构造在底层有什么不同？

## 下一步

掌握 RAII + 智能指针 + 移动语义后，进 **Stage 3: 面向对象深入**（虚函数表、菱形继承），那是 C++ OOP 的"深水区"。
