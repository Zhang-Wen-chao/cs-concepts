/**
 * 移动语义实践示例
 * 编译：g++ -std=c++17 04_move_semantics.cpp -o move_semantics
 * 运行：./move_semantics
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <algorithm>

// ============ 示例 1：拷贝 vs 移动的性能差异 ============

void test_copy_vs_move() {
    std::cout << "\n=== 示例 1: 拷贝 vs 移动的性能差异 ===" << std::endl;

    // 创建一个大 vector
    std::vector<int> large_vec(1000000, 42);
    std::cout << "创建了包含 100 万个元素的 vector" << std::endl;

    // 拷贝（慢）
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> copied = large_vec;
    auto end = std::chrono::high_resolution_clock::now();
    auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "拷贝耗时: " << copy_time.count() << " 微秒" << std::endl;

    // 移动（快）
    start = std::chrono::high_resolution_clock::now();
    std::vector<int> moved = std::move(copied);
    end = std::chrono::high_resolution_clock::now();
    auto move_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "移动耗时: " << move_time.count() << " 微秒" << std::endl;

    std::cout << "移动比拷贝快: " << (double)copy_time.count() / move_time.count() << " 倍" << std::endl;

    std::cout << "\n移动后的状态：" << std::endl;
    std::cout << "copied.size() = " << copied.size() << " (被掏空了)" << std::endl;
    std::cout << "moved.size() = " << moved.size() << std::endl;
}

// ============ 示例 2：自定义类的移动语义 ============

class MyVector {
    int* data_;
    size_t size_;

public:
    // 构造函数
    MyVector(size_t size) : size_(size) {
        data_ = new int[size];
        std::fill(data_, data_ + size, 0);
        std::cout << "  构造: 分配 " << size << " 个元素" << std::endl;
    }

    // 析构函数
    ~MyVector() {
        delete[] data_;
        // std::cout << "  析构: 释放内存" << std::endl;
    }

    // 拷贝构造函数
    MyVector(const MyVector& other) : size_(other.size_) {
        data_ = new int[size_];
        std::copy(other.data_, other.data_ + size_, data_);
        std::cout << "  拷贝构造: 复制 " << size_ << " 个元素（慢）" << std::endl;
    }

    // 拷贝赋值运算符
    MyVector& operator=(const MyVector& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
            std::cout << "  拷贝赋值: 复制 " << size_ << " 个元素（慢）" << std::endl;
        }
        return *this;
    }

    // 移动构造函数
    MyVector(MyVector&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        // "偷"走资源
        other.data_ = nullptr;
        other.size_ = 0;
        std::cout << "  移动构造: 转移所有权（O(1)，快）" << std::endl;
    }

    // 移动赋值运算符
    MyVector& operator=(MyVector&& other) noexcept {
        if (this != &other) {
            delete[] data_;  // 释放旧资源
            // "偷"走资源
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
            std::cout << "  移动赋值: 转移所有权（O(1)，快）" << std::endl;
        }
        return *this;
    }

    size_t size() const { return size_; }
    int* data() { return data_; }
};

void test_custom_move() {
    std::cout << "\n=== 示例 2: 自定义类的移动语义 ===" << std::endl;

    std::cout << "\n创建 v1:" << std::endl;
    MyVector v1(100);

    std::cout << "\n拷贝构造 v2:" << std::endl;
    MyVector v2 = v1;  // 拷贝构造

    std::cout << "\n移动构造 v3:" << std::endl;
    MyVector v3 = std::move(v1);  // 移动构造

    std::cout << "\nv1.size() = " << v1.size() << " (被掏空)" << std::endl;
    std::cout << "v2.size() = " << v2.size() << std::endl;
    std::cout << "v3.size() = " << v3.size() << std::endl;

    std::cout << "\n移动赋值 v2 = std::move(v3):" << std::endl;
    v2 = std::move(v3);  // 移动赋值

    std::cout << "\nv2.size() = " << v2.size() << std::endl;
    std::cout << "v3.size() = " << v3.size() << " (被掏空)" << std::endl;
}

// ============ 示例 3：返回值优化（RVO）============

std::vector<int> create_vector_no_move() {
    std::vector<int> vec(1000000, 42);
    return vec;  // ✅ RVO 或移动，不要加 std::move
}

std::vector<int> create_vector_with_move() {
    std::vector<int> vec(1000000, 42);
    return std::move(vec);  // ❌ 妨碍 RVO
}

void test_rvo() {
    std::cout << "\n=== 示例 3: 返回值优化（RVO）===" << std::endl;

    std::cout << "\n不加 std::move（正确，编译器优化）：" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto v1 = create_vector_no_move();
    auto end = std::chrono::high_resolution_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << time1.count() << " 微秒" << std::endl;

    std::cout << "\n加 std::move（错误，妨碍优化）：" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto v2 = create_vector_with_move();
    end = std::chrono::high_resolution_clock::now();
    auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "耗时: " << time2.count() << " 微秒" << std::endl;

    std::cout << "\n结论：返回语句不要加 std::move" << std::endl;
}

// ============ 示例 4：左值和右值 ============

void process(std::string& s) {
    std::cout << "  处理左值: " << s << std::endl;
}

void process(std::string&& s) {
    std::cout << "  处理右值: " << s << std::endl;
}

void test_lvalue_rvalue() {
    std::cout << "\n=== 示例 4: 左值和右值 ===" << std::endl;

    std::string s1 = "hello";

    std::cout << "\n传左值：" << std::endl;
    process(s1);  // 调用 process(string&)

    std::cout << "\n传右值（临时对象）：" << std::endl;
    process("world");  // 调用 process(string&&)

    std::cout << "\n传右值（std::move）：" << std::endl;
    process(std::move(s1));  // 调用 process(string&&)

    std::cout << "\n注意：s1 被移动后不要再用！" << std::endl;
}

// ============ 示例 5：unique_ptr 只能移动 ============

void test_unique_ptr_move() {
    std::cout << "\n=== 示例 5: unique_ptr 只能移动 ===" << std::endl;

    auto p1 = std::make_unique<int>(42);
    std::cout << "p1 值: " << *p1 << std::endl;

    // auto p2 = p1;  // ❌ 编译错误：不能拷贝

    auto p2 = std::move(p1);  // ✅ 移动，转移所有权
    std::cout << "p2 值: " << *p2 << std::endl;

    if (!p1) {
        std::cout << "p1 现在是空的" << std::endl;
    }

    std::cout << "\n这就是移动语义的应用：独占所有权" << std::endl;
}

// ============ 示例 6：容器中的移动 ============

void test_vector_move() {
    std::cout << "\n=== 示例 6: 容器中的移动 ===" << std::endl;

    std::vector<std::string> vec;

    std::string s1 = "short";
    std::string s2 = "this is a very long string that will demonstrate the performance difference";

    std::cout << "\npush_back 拷贝：" << std::endl;
    vec.push_back(s1);  // 拷贝（s1 还要用）
    std::cout << "s1 = \"" << s1 << "\"（还在）" << std::endl;

    std::cout << "\npush_back 移动：" << std::endl;
    vec.push_back(std::move(s2));  // 移动（s2 不再用）
    std::cout << "s2 = \"" << s2 << "\"（被掏空）" << std::endl;

    std::cout << "\nvec 内容：" << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "  [" << i << "] = \"" << vec[i] << "\"" << std::endl;
    }
}

// ============ 示例 7：移动后不能用 ============

void test_moved_from_object() {
    std::cout << "\n=== 示例 7: 移动后不能用 ===" << std::endl;

    std::string s1 = "hello world";
    std::cout << "s1 = \"" << s1 << "\"" << std::endl;

    std::string s2 = std::move(s1);  // 移动
    std::cout << "s2 = \"" << s2 << "\"" << std::endl;

    std::cout << "\ns1 被移动后的状态：" << std::endl;
    std::cout << "s1.empty() = " << (s1.empty() ? "true" : "false") << std::endl;
    std::cout << "s1.size() = " << s1.size() << std::endl;

    std::cout << "\n注意：" << std::endl;
    std::cout << "  - s1 现在处于\"有效但未指定\"的状态" << std::endl;
    std::cout << "  - 可以：赋新值、销毁" << std::endl;
    std::cout << "  - 不可以：使用其内容" << std::endl;

    s1 = "new value";  // ✅ 可以重新赋值
    std::cout << "\n重新赋值后 s1 = \"" << s1 << "\"" << std::endl;
}

// ============ 示例 8：const 对象不能移动 ============

void test_const_object() {
    std::cout << "\n=== 示例 8: const 对象不能移动 ===" << std::endl;

    const std::string s1 = "hello";
    std::cout << "const string s1 = \"" << s1 << "\"" << std::endl;

    std::cout << "\nstd::move(s1) 实际会调用拷贝：" << std::endl;
    std::string s2 = std::move(s1);  // 实际调用拷贝构造

    std::cout << "s1 = \"" << s1 << "\"（没有被掏空）" << std::endl;
    std::cout << "s2 = \"" << s2 << "\"" << std::endl;

    std::cout << "\n原因：const 对象不能修改，所以不能\"掏空\"" << std::endl;
}

// ============ 示例 9：std::swap 的移动实现 ============

template<typename T>
void my_swap(T& a, T& b) {
    T temp = std::move(a);    // 移动
    a = std::move(b);         // 移动
    b = std::move(temp);      // 移动
    // 三次移动，而不是三次拷贝
}

void test_swap() {
    std::cout << "\n=== 示例 9: swap 的移动实现 ===" << std::endl;

    std::string s1 = "string 1";
    std::string s2 = "string 2";

    std::cout << "交换前：" << std::endl;
    std::cout << "s1 = \"" << s1 << "\"" << std::endl;
    std::cout << "s2 = \"" << s2 << "\"" << std::endl;

    my_swap(s1, s2);  // 用移动实现的 swap

    std::cout << "\n交换后：" << std::endl;
    std::cout << "s1 = \"" << s1 << "\"" << std::endl;
    std::cout << "s2 = \"" << s2 << "\"" << std::endl;

    std::cout << "\n旧实现：三次拷贝（慢）" << std::endl;
    std::cout << "新实现：三次移动（快）" << std::endl;
}

// ============ 示例 10：Rule of Zero vs Rule of Five ============

// Rule of Zero（推荐）
class GoodClass {
    std::string name_;
    std::vector<int> data_;
    std::unique_ptr<int> ptr_;

public:
    GoodClass(const std::string& name, size_t size)
        : name_(name), data_(size), ptr_(std::make_unique<int>(42)) {}

    // 编译器自动生成：
    // - 析构函数
    // - 拷贝构造/赋值
    // - 移动构造/赋值
};

// Rule of Five（自己管理资源时）
class ManualClass {
    int* data_;
    size_t size_;

public:
    ManualClass(size_t size) : size_(size) {
        data_ = new int[size];
    }

    ~ManualClass() {
        delete[] data_;
    }

    ManualClass(const ManualClass& other) : size_(other.size_) {
        data_ = new int[size_];
        std::copy(other.data_, other.data_ + size_, data_);
    }

    ManualClass& operator=(const ManualClass& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            data_ = new int[size_];
            std::copy(other.data_, other.data_ + size_, data_);
        }
        return *this;
    }

    ManualClass(ManualClass&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    ManualClass& operator=(ManualClass&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
};

void test_rule_of_zero() {
    std::cout << "\n=== 示例 10: Rule of Zero vs Rule of Five ===" << std::endl;

    std::cout << "\nRule of Zero（推荐）：" << std::endl;
    std::cout << "  - 用 unique_ptr、vector 等管理资源" << std::endl;
    std::cout << "  - 编译器自动生成所有特殊成员函数" << std::endl;
    std::cout << "  - 自动正确，不会出错" << std::endl;

    GoodClass g1("test", 100);
    GoodClass g2 = std::move(g1);  // 自动支持移动
    std::cout << "  GoodClass 自动支持移动" << std::endl;

    std::cout << "\nRule of Five（自己管理资源时）：" << std::endl;
    std::cout << "  - 需要自定义析构函数" << std::endl;
    std::cout << "  - 通常需要自定义所有五个特殊成员函数" << std::endl;
    std::cout << "  - 容易出错，不推荐" << std::endl;

    ManualClass m1(100);
    ManualClass m2 = std::move(m1);  // 手动实现的移动
    std::cout << "  ManualClass 手动实现移动" << std::endl;
}

// ============ 示例 11：性能对比总结 ============

void print_performance_summary() {
    std::cout << "\n=== 示例 11: 性能对比总结 ===" << std::endl;

    std::cout << "\n操作            时间复杂度    性能" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "RVO/NRVO        O(0)          最快（零开销）" << std::endl;
    std::cout << "移动            O(1)          快" << std::endl;
    std::cout << "拷贝            O(n)          慢" << std::endl;

    std::cout << "\n优化策略：" << std::endl;
    std::cout << "1. RVO/NRVO > 移动 > 拷贝" << std::endl;
    std::cout << "2. 返回值让编译器优化，不要手动 move" << std::endl;
    std::cout << "3. 移动后不要使用原对象" << std::endl;
    std::cout << "4. 移动构造函数要 noexcept" << std::endl;
    std::cout << "5. 用 Rule of Zero（让标准库管理资源）" << std::endl;
}

// ============ 主函数 ============

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      移动语义实践示例" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_copy_vs_move();
        test_custom_move();
        test_rvo();
        test_lvalue_rvalue();
        test_unique_ptr_move();
        test_vector_move();
        test_moved_from_object();
        test_const_object();
        test_swap();
        test_rule_of_zero();
        print_performance_summary();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  所有示例运行完成！✅" << std::endl;
        std::cout << "========================================" << std::endl;

        std::cout << "\n关键收获：" << std::endl;
        std::cout << "1. 移动 = 转移所有权，不拷贝数据（O(1)）" << std::endl;
        std::cout << "2. 移动比拷贝快得多（通常几千倍）" << std::endl;
        std::cout << "3. 返回值让编译器优化，不要 std::move" << std::endl;
        std::cout << "4. 移动后不要使用原对象" << std::endl;
        std::cout << "5. const 对象不能移动" << std::endl;
        std::cout << "6. 移动构造函数要 noexcept" << std::endl;
        std::cout << "7. 用 Rule of Zero（让标准库管理资源）" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
