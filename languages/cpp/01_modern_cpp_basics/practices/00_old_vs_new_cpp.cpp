/**
 * 新旧 C++ 对比示例
 * 展示从 C++98 到现代 C++ 的思维转变
 *
 * 编译：g++ -std=c++17 00_old_vs_new_cpp.cpp -o old_vs_new
 * 运行：./old_vs_new
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <chrono>
#include <unordered_map>

// ============ 示例 1：内存管理 ============

void test_memory_management() {
    std::cout << "\n=== 示例 1: 内存管理 ===" << std::endl;

    // ❌ 旧风格（C++98）：手动管理，容易出错
    std::cout << "\n旧风格（手动 new/delete）：" << std::endl;
    {
        int* data = new int[1000];
        std::cout << "手动分配内存..." << std::endl;

        // ... 使用数据 ...
        data[0] = 42;

        delete[] data;  // 容易忘记！
        std::cout << "手动释放内存" << std::endl;
    }

    // ✅ 新风格（C++11+）：自动管理
    std::cout << "\n新风格（自动管理）：" << std::endl;
    {
        std::vector<int> data(1000);
        std::cout << "自动分配内存..." << std::endl;

        // ... 使用数据 ...
        data[0] = 42;

        std::cout << "离开作用域，自动释放内存" << std::endl;
    }  // 自动释放，不会忘记
}

// ============ 示例 2：文件处理 ============

void test_file_handling() {
    std::cout << "\n=== 示例 2: 文件处理 ===" << std::endl;

    // 先创建测试文件
    {
        std::ofstream out("test_data.txt");
        out << "Line 1\nLine 2\nLine 3\n";
    }

    // ❌ 旧风格（C++98）：手动管理文件
    std::cout << "\n旧风格（手动 fopen/fclose）：" << std::endl;
    {
        FILE* f = std::fopen("test_data.txt", "r");
        if (!f) {
            std::cout << "Failed to open file" << std::endl;
            return;
        }

        char buffer[256];
        while (std::fgets(buffer, sizeof(buffer), f)) {
            // 处理数据
        }

        std::fclose(f);  // 容易忘记，或者异常时跳过
        std::cout << "手动关闭文件" << std::endl;
    }

    // ✅ 新风格（C++11+）：RAII 自动管理
    std::cout << "\n新风格（RAII，自动管理）：" << std::endl;
    {
        std::ifstream file("test_data.txt");
        if (!file) {
            std::cout << "Failed to open file" << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            // 处理数据
        }

        std::cout << "离开作用域，自动关闭文件" << std::endl;
    }  // 自动关闭，即使有异常
}

// ============ 示例 3：容器管理 ============

// 旧风格：手写数组类
class OldIntArray {
    int* data_;
    size_t size_;

public:
    OldIntArray(size_t size) : size_(size) {
        data_ = new int[size];
        std::cout << "OldIntArray: 手动分配 " << size << " 个元素" << std::endl;
    }

    ~OldIntArray() {
        delete[] data_;
        std::cout << "OldIntArray: 手动释放" << std::endl;
    }

    // 还需要实现拷贝构造、赋值运算符等...
    // 很容易忘记，导致浅拷贝问题

    int& operator[](size_t i) { return data_[i]; }
    size_t size() const { return size_; }
};

void test_container() {
    std::cout << "\n=== 示例 3: 容器管理 ===" << std::endl;

    // ❌ 旧风格：手写容器
    std::cout << "\n旧风格（手写容器）：" << std::endl;
    {
        OldIntArray arr(100);
        arr[0] = 42;
        // 需要自己管理内存
    }  // 析构时释放

    // ✅ 新风格：使用标准库
    std::cout << "\n新风格（标准库容器）：" << std::endl;
    {
        std::vector<int> vec(100);
        std::cout << "std::vector: 自动管理 " << vec.size() << " 个元素" << std::endl;
        vec[0] = 42;
        // 自动管理内存，拷贝、移动都自动处理
    }
    std::cout << "std::vector: 自动释放" << std::endl;
}

// ============ 示例 4：智能指针 vs 裸指针 ============

void test_smart_pointers() {
    std::cout << "\n=== 示例 4: 智能指针 vs 裸指针 ===" << std::endl;

    // ❌ 旧风格：裸指针
    std::cout << "\n旧风格（裸指针）：" << std::endl;
    {
        int* p = new int(42);
        std::cout << "裸指针：值 = " << *p << std::endl;
        std::cout << "问题：谁负责 delete？什么时候 delete？" << std::endl;
        delete p;  // 容易忘记
    }

    // ✅ 新风格：智能指针
    std::cout << "\n新风格（智能指针）：" << std::endl;
    {
        // unique_ptr：独占所有权
        auto p1 = std::make_unique<int>(42);
        std::cout << "unique_ptr：值 = " << *p1 << std::endl;
        std::cout << "所有权清晰，自动释放" << std::endl;

        // shared_ptr：共享所有权
        auto p2 = std::make_shared<int>(100);
        auto p3 = p2;  // 引用计数 +1
        std::cout << "shared_ptr：引用计数 = " << p2.use_count() << std::endl;
    }  // 自动释放，不会泄漏
    std::cout << "智能指针：自动释放" << std::endl;
}

// ============ 示例 5：移动语义 ============

void test_move_semantics() {
    std::cout << "\n=== 示例 5: 移动语义（性能优化）===" << std::endl;

    auto create_large_vector = []() {
        std::vector<int> vec(1000000, 42);
        std::cout << "创建了包含 100 万个元素的 vector" << std::endl;
        return vec;  // 现代 C++：移动，不拷贝（O(1)）
    };

    std::cout << "\n旧 C++98：会拷贝（慢）" << std::endl;
    std::cout << "新 C++11+：自动移动（快）" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> v = create_large_vector();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "返回耗时: " << duration.count() << " 微秒" << std::endl;
    std::cout << "（如果是拷贝，会慢得多！）" << std::endl;
}

// ============ 示例 6：Lambda 表达式 ============

void test_lambda() {
    std::cout << "\n=== 示例 6: Lambda 表达式 ===" << std::endl;

    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // ❌ 旧风格：定义命名函数
    std::cout << "\n旧风格（命名函数）：" << std::endl;
    struct {
        bool operator()(int x) const { return x % 2 == 0; }
    } is_even_functor;

    int even_count_old = std::count_if(numbers.begin(), numbers.end(), is_even_functor);
    std::cout << "偶数个数: " << even_count_old << std::endl;

    // ✅ 新风格：Lambda
    std::cout << "\n新风格（Lambda 表达式）：" << std::endl;
    int even_count_new = std::count_if(numbers.begin(), numbers.end(),
                                       [](int x) { return x % 2 == 0; });
    std::cout << "偶数个数: " << even_count_new << std::endl;

    // Lambda 捕获外部变量
    int threshold = 5;
    auto count_large = std::count_if(numbers.begin(), numbers.end(),
                                     [threshold](int x) { return x > threshold; });
    std::cout << "大于 " << threshold << " 的数: " << count_large << std::endl;
}

// ============ 示例 7：类型推导（auto）============

void test_auto() {
    std::cout << "\n=== 示例 7: 类型推导（auto）===" << std::endl;

    // ❌ 旧风格：写完整类型
    std::cout << "\n旧风格（完整类型名）：" << std::endl;
    std::unordered_map<std::string, std::vector<int>> old_map;
    old_map["numbers"] = {1, 2, 3};
    std::unordered_map<std::string, std::vector<int>>::iterator old_it = old_map.begin();
    std::cout << "类型名很长！" << std::endl;

    // ✅ 新风格：auto
    std::cout << "\n新风格（auto 推导）：" << std::endl;
    std::unordered_map<std::string, std::vector<int>> new_map;
    new_map["numbers"] = {1, 2, 3};
    auto new_it = new_map.begin();  // 编译器推导类型
    std::cout << "简洁清晰！" << std::endl;

    // C++17 结构化绑定
    std::cout << "\nC++17 结构化绑定：" << std::endl;
    for (const auto& [key, value] : new_map) {
        std::cout << "Key: " << key << ", Size: " << value.size() << std::endl;
    }
}

// ============ 示例 8：const 正确性 ============

void read_only(const std::string& str) {
    std::cout << "只读: " << str << std::endl;
    // str[0] = 'x';  // ❌ 编译错误：不能修改 const
}

void modify(std::string& str) {
    str[0] = 'X';
    std::cout << "修改后: " << str << std::endl;
}

void test_const_correctness() {
    std::cout << "\n=== 示例 8: const 正确性 ===" << std::endl;

    std::string text = "hello";

    std::cout << "\nconst 引用（不会修改）：" << std::endl;
    read_only(text);
    std::cout << "原始字符串: " << text << std::endl;

    std::cout << "\n非 const 引用（会修改）：" << std::endl;
    modify(text);
    std::cout << "修改后字符串: " << text << std::endl;
}

// ============ 示例 9：异常安全 ============

void test_exception_safety() {
    std::cout << "\n=== 示例 9: 异常安全 ===" << std::endl;

    // ❌ 旧风格：不异常安全
    std::cout << "\n旧风格（不安全）：" << std::endl;
    std::cout << "如果用裸指针，异常会导致内存泄漏" << std::endl;

    // ✅ 新风格：异常安全
    std::cout << "\n新风格（RAII 异常安全）：" << std::endl;
    try {
        std::vector<int> data(100);
        std::cout << "创建 vector" << std::endl;

        // 模拟异常
        throw std::runtime_error("模拟异常");

    } catch (const std::exception& e) {
        std::cout << "捕获异常: " << e.what() << std::endl;
        std::cout << "vector 自动清理，不会泄漏！" << std::endl;
    }
}

// ============ 示例 10：总结对比 ============

void print_summary() {
    std::cout << "\n=== 总结：新旧 C++ 对比 ===" << std::endl;
    std::cout << "\n旧 C++98 的问题：" << std::endl;
    std::cout << "  ❌ 手动管理内存（new/delete）" << std::endl;
    std::cout << "  ❌ 容易忘记释放资源" << std::endl;
    std::cout << "  ❌ 不异常安全" << std::endl;
    std::cout << "  ❌ 代码冗长" << std::endl;
    std::cout << "  ❌ 性能差（不必要的拷贝）" << std::endl;

    std::cout << "\n现代 C++ 的优势：" << std::endl;
    std::cout << "  ✅ RAII：自动管理资源" << std::endl;
    std::cout << "  ✅ 智能指针：不会泄漏" << std::endl;
    std::cout << "  ✅ 标准库：久经考验" << std::endl;
    std::cout << "  ✅ 移动语义：高性能" << std::endl;
    std::cout << "  ✅ Lambda：代码简洁" << std::endl;
    std::cout << "  ✅ auto：类型推导" << std::endl;
    std::cout << "  ✅ 异常安全：可靠性高" << std::endl;

    std::cout << "\n核心原则：" << std::endl;
    std::cout << "  🎯 让编译器帮你管理资源" << std::endl;
    std::cout << "  🎯 用 RAII 绑定资源生命周期" << std::endl;
    std::cout << "  🎯 用标准库，不重复造轮子" << std::endl;
    std::cout << "  🎯 const 正确性，意图明确" << std::endl;
}

// ============ 主函数 ============

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  现代 C++ vs 旧 C++ 对比示例" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_memory_management();
        test_file_handling();
        test_container();
        test_smart_pointers();
        test_move_semantics();
        test_lambda();
        test_auto();
        test_const_correctness();
        test_exception_safety();
        print_summary();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  所有示例运行完成！✅" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
