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

**Rule of 0（推荐）**：能用标准库就什么都不写
```cpp
class Good {
    std::vector<int> data_;        // 自动管理
    std::unique_ptr<int> ptr_;     // 自动管理
    // 不需要写任何特殊成员函数
};
```

**Rule of 5**：自己管理资源时，必须定义 5 个函数
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

## 标准库 RAII 类

```cpp
// 内存管理
std::unique_ptr<int> p = std::make_unique<int>(10);
std::vector<int> vec(1000);

// 文件管理
std::ifstream file("data.txt");  // 自动打开和关闭

// 锁管理
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);  // 自动加锁
    // 临界区
}  // 自动解锁
```

## 关键原则

1. **构造时获取资源，析构时释放资源**
2. **析构函数不抛异常**（标记 `noexcept`）
3. **优先用标准库**（Rule of 0）
4. **禁止拷贝或正确实现拷贝**
5. **移动构造函数要 `noexcept`**

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
