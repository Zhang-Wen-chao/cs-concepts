# RAII 原则

> Resource Acquisition Is Initialization（资源生命周期绑定对象生命周期）

## 核心原理

**C++ 保证：对象离开作用域时，一定调用析构函数**

```cpp
void foo() {
    std::vector<int> vec(1000);  // 构造时获取资源
    // ...
}  // 离开作用域，自动调用析构释放资源，即使有异常
```

## 自己实现 RAII 类

```cpp
class FileHandle {
    FILE* file_;
public:
    // === 1. 构造函数：打开文件（RAII 的"获取资源"）===
    FileHandle(const std::string& name) : file_(fopen(name.c_str(), "r")) {
        if (!file_) throw std::runtime_error("Cannot open file");
    }
    /* 符号说明：
     * - FileHandle(...)：构造函数，与类同名
     * - const std::string& name：参数
     *   · const：不修改参数
     *   · std::string：C++ 字符串类型
     *   · &：引用传递，避免拷贝
     *   · name：参数名
     * - :：成员初始化列表开始
     * - file_(...)：初始化成员变量 file_
     * - fopen(name.c_str(), "r")：C 函数打开文件
     *   · name.c_str()：转为 C 字符串
     *   · "r"：只读模式
     * - if (!file_)：检查是否打开失败（! 是逻辑非）
     * - throw：抛出异常
     * - std::runtime_error：标准运行时错误类型
     *
     * 整体含义：构造时打开文件，失败则抛异常
     */

    // === 2. 析构函数：关闭文件（RAII 的"释放资源"）===
    ~FileHandle() { if (file_) fclose(file_); }
    /* 符号说明：
     * - ~FileHandle()：析构函数（~ 表示析构）
     * - if (file_)：检查指针是否有效（不为 nullptr）
     * - fclose(file_)：C 函数关闭文件
     *
     * 整体含义：对象销毁时自动关闭文件
     */

    // === 3. 禁止拷贝（防止两个对象管理同一文件）===
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    /* 符号说明：
     * - FileHandle(const FileHandle&)：拷贝构造函数签名
     *   · const FileHandle&：常量引用参数（拷贝源）
     * - FileHandle& operator=(const FileHandle&)：拷贝赋值运算符
     *   · FileHandle&：返回引用（支持链式赋值 a=b=c）
     *   · operator=：重载赋值运算符 =
     * - = delete：C++11 特殊语法，删除该函数
     *   · 注意：这不是"给函数赋值"，而是特殊的声明形式
     *   · 作用：告诉编译器"这个函数存在，但禁止调用"
     *   · 效果：如果有代码试图拷贝，编译时报错
     *
     * 为什么要禁止拷贝？看这个例子：
     *
     *   FileHandle f1("test.txt");  // f1.file_ 指向文件
     *   FileHandle f2 = f1;         // 如果允许拷贝，f2.file_ 也指向同一文件
     *   // 问题：现在 f1 和 f2 都管理同一个 file_
     *   // 当 f1 析构 → fclose(file_)  ✓ 关闭文件
     *   // 当 f2 析构 → fclose(file_)  ✗ 重复关闭！崩溃！
     *
     * 使用 = delete 后：
     *   FileHandle f1("test.txt");
     *   FileHandle f2 = f1;  // 编译错误："拷贝构造函数已被删除"
     *
     * 整体含义：明确禁止拷贝，编译期阻止资源重复管理，避免重复释放导致崩溃
     */

    // === 4. 移动构造：转移文件所有权 ===
    FileHandle(FileHandle&& o) noexcept : file_(o.file_) {
        o.file_ = nullptr;
    }
    /* 符号说明：
     * - FileHandle&&：右值引用类型
     *   · &&：右值引用（接收临时对象）
     *   · o：参数名（other 的缩写）
     *
     * 什么是右值引用（&&）？
     *   · 普通引用 T&：只能绑定到有名字的对象（左值）
     *     例如：FileHandle f1; FileHandle& ref = f1; ✓
     *   · 右值引用 T&&：能绑定到临时对象（右值）
     *     例如：FileHandle&& ref = createFile(); ✓ 临时对象
     *   · 作用：允许"接管"临时对象的资源，而不是拷贝
     *
     * 为什么需要移动？看对比：
     *
     *   拷贝方式（昂贵）：
     *     FileHandle f1("test.txt");
     *     FileHandle f2 = f1;  // 需要重新打开文件，拷贝所有数据
     *
     *   移动方式（高效）：
     *     FileHandle f1("test.txt");
     *     FileHandle f2 = std::move(f1);  // 只是转移指针，不重新打开
     *     // f1.file_ 变成 nullptr，f2.file_ 接管文件
     *
     * - noexcept：承诺不抛异常（移动操作必须标记）
     *   · 为什么必须？vector 等容器在扩容时会检查 noexcept
     *   · 如果移动可能抛异常，容器会退回到拷贝（性能差）
     *
     * - : file_(o.file_)：初始化列表，直接接管 o 的文件指针
     * - o.file_ = nullptr：将原对象指针置空
     *   · nullptr：C++11 空指针常量
     *   · 必须置空！否则 o 析构时会关闭文件，导致 this 的 file_ 失效
     *
     * 整体含义：从临时对象"搬走"资源（转移指针），而不是复制，避免昂贵的拷贝
     */

    // === 5. 移动赋值：转移文件所有权（赋值形式）===
    FileHandle& operator=(FileHandle&& o) noexcept {
        if (this != &o) {
            if (file_) fclose(file_);
            file_ = o.file_;
            o.file_ = nullptr;
        }
        return *this;
    }
    /* 符号说明：
     * - FileHandle& operator=(FileHandle&& o)：移动赋值运算符
     *   · 返回 FileHandle&：支持链式赋值
     *   · operator=：重载赋值运算符
     *   · FileHandle&&：右值引用参数
     * - noexcept：不抛异常
     * - if (this != &o)：检查自我赋值
     *   · this：当前对象指针
     *   · &o：取 o 的地址
     *   · !=：不等于运算符
     * - if (file_) fclose(file_)：先关闭当前文件
     * - file_ = o.file_：接管新文件
     * - o.file_ = nullptr：置空原对象
     * - return *this：返回当前对象引用
     *   · *this：解引用 this 指针
     *
     * 整体含义：通过赋值转移所有权，先释放当前资源再接管新资源
     */
};
```

## Rule of 0/3/5

> **命名说明**：这个名字包含了三个不同时代的规则
> - **Rule of 3**：C++98/03 时代（拷贝时代）
> - **Rule of 5**：C++11 引入移动语义后
> - **Rule of 0**：现代 C++ 最佳实践
>
> **实际使用**：只需记住 **Rule of 0 or 5**
> - 要么用标准库（0 个特殊函数）
> - 要么自己管理资源（5 个全写）
> - **Rule of 3 已过时，仅供理解老代码**

### Rule of 0（推荐）✅

**能用标准库就什么都不写**

```cpp
class Good {
    std::vector<int> data_;        // 自动管理
    std::unique_ptr<int> ptr_;     // 自动管理
    // 不需要写任何特殊成员函数
};
```

**为什么推荐？**
- 标准库已经正确实现了资源管理
- 不会写错
- 代码简洁

### Rule of 5（自己管理资源时）

**如果直接持有裸指针/文件句柄等资源，必须定义 5 个函数**

```cpp
class MyResource {
public:
    ~MyResource();                              // 1. 析构
    MyResource(const MyResource&);              // 2. 拷贝构造
    MyResource& operator=(const MyResource&);   // 3. 拷贝赋值
    MyResource(MyResource&&) noexcept;          // 4. 移动构造
    MyResource& operator=(MyResource&&) noexcept; // 5. 移动赋值
};
```

**为什么是 5 个？**
- 只写析构 → 拷贝时浅拷贝 → 重复释放崩溃
- 不写移动 → 退化为拷贝 → 性能差

### Rule of 3（已过时，仅供理解）

**C++11 之前没有移动语义，只需 3 个函数**

```cpp
class OldStyle {
    ~OldStyle();                           // 1. 析构
    OldStyle(const OldStyle&);             // 2. 拷贝构造
    OldStyle& operator=(const OldStyle&);  // 3. 拷贝赋值
};
```

**现代 C++ 不要用！** 因为缺少移动函数会导致性能问题。

## 标准库 RAII 类

> **标准库几乎所有类都是 RAII 的！** 以下是常见示例：

```cpp
// === 1. 内存管理 ===
std::unique_ptr<int> p = std::make_unique<int>(10);  // 独占所有权
std::shared_ptr<int> sp = std::make_shared<int>(20); // 共享所有权
std::vector<int> vec(1000);                          // 动态数组
std::string str = "hello";                           // 字符串
std::array<int, 5> arr;                              // 固定大小数组（栈上）
std::deque<int> deq;                                 // 双端队列
std::list<int> lst;                                  // 链表
std::map<int, std::string> m;                        // 映射
std::set<int> s;                                     // 集合

// === 2. 文件/流管理 ===
std::ifstream file("data.txt");     // 输入文件流（自动关闭）
std::ofstream out("output.txt");    // 输出文件流
std::fstream fs("file.txt");        // 读写文件流
std::stringstream ss;               // 字符串流

// === 3. 线程同步（锁管理）===
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);        // 自动加锁/解锁
    std::unique_lock<std::mutex> ulock(mtx);      // 更灵活的锁
    std::shared_lock<std::shared_mutex> slock(m); // 读锁（C++17）
    std::scoped_lock lock(mtx1, mtx2);            // 多锁（C++17）
}

// === 4. 线程管理 ===
std::thread t([]{ /* work */ });  // 线程对象
t.join();  // 必须 join 或 detach，否则析构时崩溃

std::jthread jt([]{ /* work */ }); // C++20，自动 join

// === 5. 异常安全的资源管理 ===
std::optional<int> opt = 42;            // 可能有值
std::variant<int, std::string> var = 5; // 类型安全的 union

// === 6. 函数对象 ===
std::function<int(int)> f = [](int x) { return x * 2; };

// === 7. 正则表达式 ===
std::regex pattern(R"(\d+)");

// === 8. 时间管理 ===
auto start = std::chrono::steady_clock::now();
// ... 代码 ...
auto end = std::chrono::steady_clock::now();
```

**核心原则**：标准库的类型都遵循 RAII，构造时分配资源，析构时自动释放

## 关键原则

1. **构造时获取资源，析构时释放资源**
2. **析构函数不抛异常**（标记 `noexcept`）
3. **优先用标准库**（Rule of 0）
4. **禁止拷贝或正确实现拷贝**
5. **移动构造函数要 `noexcept`**

### 关于 `noexcept` 的重要说明

> **⚠️ `noexcept` 是程序员的承诺，不是编译器的检查！**

```cpp
// 编译能通过，但运行时会崩溃
void foo() noexcept {
    throw std::runtime_error("oops");  // ✅ 编译通过
    // ❌ 运行时调用 std::terminate，程序直接终止（不是异常）
}
```

**为什么要标记 `noexcept`？**

1. **让容器敢用移动而不是拷贝**
   ```cpp
   class MyClass {
       MyClass(MyClass&&) noexcept;  // 有 noexcept
   };

   std::vector<MyClass> vec;
   vec.push_back(...);  // 扩容时用移动（快）✅
   ```

   ```cpp
   class Slow {
       Slow(Slow&&);  // 没有 noexcept
   };

   std::vector<Slow> vec;
   vec.push_back(...);  // 扩容时用拷贝（慢）❌
   ```

   **原因**：如果移动可能抛异常，`vector` 扩容到一半失败会导致数据损坏，所以只能用安全但慢的拷贝。

2. **析构函数默认就是 `noexcept`**
   - 如果在异常处理时析构又抛异常 → 两个异常同时存在 → C++ 无法处理 → 程序终止
   - 所以析构函数**绝对不能**抛异常

**如何保证不抛异常？**

```cpp
// ✅ 安全的操作
delete ptr;           // 不抛异常
fclose(file);         // 不抛异常
ptr = nullptr;        // 不抛异常
基本类型赋值          // 不抛异常

// ❌ 危险的操作（不要在 noexcept 函数里做）
new int;              // 内存不足会抛异常
vec.push_back();      // 扩容失败会抛异常
throw ...;            // 显式抛出异常
```

**记住**：在析构和移动函数里只做简单的指针/资源操作，自然就不会抛异常。

## RAII vs 手动管理

```cpp
// ❌ 手动管理：容易出错
void bad() {
    int* p = new int[1000];
    if (error) return;  // 内存泄漏
    delete[] p;
}

// ✅ RAII：不会出错
void good() {
    std::vector<int> vec(1000);
    if (error) return;  // 自动释放
}
```
