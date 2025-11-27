/**
 * RAII 实践示例
 * 编译：g++ -std=c++17 01_raii_examples.cpp -o raii_examples
 * 运行：./raii_examples
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <chrono>
#include <thread>
#include <cstdio>
#include <stdexcept>

// ============ 示例 1：文件管理 ============

class FileHandle {
public:
    explicit FileHandle(const std::string& filename, const char* mode = "r") {
        file_ = std::fopen(filename.c_str(), mode);
        if (!file_) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        std::cout << "File opened: " << filename << std::endl;
    }

    ~FileHandle() {
        if (file_) {
            std::fclose(file_);
            std::cout << "File closed" << std::endl;
        }
    }

    // 禁止拷贝
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // 允许移动
    FileHandle(FileHandle&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }

    FILE* get() const { return file_; }

private:
    FILE* file_ = nullptr;
};

void test_file_raii() {
    std::cout << "\n=== Test 1: File RAII ===" << std::endl;

    try {
        // 创建测试文件
        {
            std::ofstream out("test.txt");
            out << "Hello RAII\n";
        }

        // 使用 RAII 读取文件
        {
            FileHandle file("test.txt");
            char buffer[100];
            if (std::fgets(buffer, sizeof(buffer), file.get())) {
                std::cout << "Read: " << buffer;
            }
        }  // 自动关闭文件

        std::cout << "File automatically closed!" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

// ============ 示例 2：锁管理 ============

std::mutex global_mutex;
int shared_counter = 0;

void test_lock_raii() {
    std::cout << "\n=== Test 2: Lock RAII ===" << std::endl;

    auto increment = [](int n) {
        for (int i = 0; i < n; ++i) {
            std::lock_guard<std::mutex> lock(global_mutex);  // RAII 锁
            ++shared_counter;
        }  // 自动解锁
    };

    // 启动多个线程
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(increment, 100);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Counter: " << shared_counter << " (expected: 1000)" << std::endl;
}

// ============ 示例 3：计时器 ============

class Timer {
public:
    explicit Timer(const std::string& name) : name_(name) {
        start_ = std::chrono::steady_clock::now();
        std::cout << name_ << " started..." << std::endl;
    }

    ~Timer() {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << name_ << " took " << duration.count() << "ms" << std::endl;
    }

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

void test_timer_raii() {
    std::cout << "\n=== Test 3: Timer RAII ===" << std::endl;

    {
        Timer timer("Slow operation");

        // 模拟耗时操作
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }  // 自动打印耗时
}

// ============ 示例 4：内存管理（手写 unique_ptr）============

template<typename T>
class UniquePtr {
public:
    explicit UniquePtr(T* ptr = nullptr) : ptr_(ptr) {
        if (ptr_) {
            std::cout << "UniquePtr acquired resource" << std::endl;
        }
    }

    ~UniquePtr() {
        if (ptr_) {
            delete ptr_;
            std::cout << "UniquePtr released resource" << std::endl;
        }
    }

    // 禁止拷贝
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    // 允许移动
    UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        std::cout << "UniquePtr moved" << std::endl;
    }

    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get() const { return ptr_; }

private:
    T* ptr_;
};

void test_unique_ptr() {
    std::cout << "\n=== Test 4: UniquePtr RAII ===" << std::endl;

    {
        UniquePtr<int> p1(new int(42));
        std::cout << "Value: " << *p1 << std::endl;

        // 移动
        UniquePtr<int> p2 = std::move(p1);
        std::cout << "After move, p2 value: " << *p2 << std::endl;

    }  // 自动释放内存
}

// ============ 示例 5：异常安全 ============

void test_exception_safety() {
    std::cout << "\n=== Test 5: Exception Safety ===" << std::endl;

    // ❌ 不安全的代码（注释掉，不运行）
    /*
    int* data = new int[1000];
    // 如果这里抛异常，内存泄漏
    throw std::runtime_error("error");
    delete[] data;  // 永远不会执行
    */

    // ✅ RAII：异常安全
    try {
        std::vector<int> data(1000);
        std::cout << "Vector created" << std::endl;

        // 模拟异常
        if (true) {
            throw std::runtime_error("Oops!");
        }

    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
        std::cout << "Vector automatically cleaned up!" << std::endl;
    }
}

// ============ 示例 6：对比手动管理 vs RAII ============

void test_manual_vs_raii() {
    std::cout << "\n=== Test 6: Manual vs RAII ===" << std::endl;

    // 手动管理：容易出错
    std::cout << "Manual management:" << std::endl;
    FILE* file = std::fopen("test.txt", "r");
    if (file) {
        // ... 处理 ...
        std::fclose(file);  // 容易忘记
        std::cout << "Manually closed file" << std::endl;
    }

    // RAII：自动管理
    std::cout << "\nRAII management:" << std::endl;
    {
        std::ifstream file("test.txt");  // 自动打开
        // ... 处理 ...
    }  // 自动关闭
    std::cout << "File automatically closed" << std::endl;
}

// ============ 示例 7：多资源管理 ============

class MultiResource {
public:
    MultiResource() {
        std::cout << "Acquiring multiple resources..." << std::endl;
        resource1_ = std::make_unique<std::vector<int>>(100);
        resource2_ = std::make_unique<std::string>("data");
        std::cout << "All resources acquired" << std::endl;
    }

    ~MultiResource() {
        std::cout << "Releasing all resources..." << std::endl;
        // unique_ptr 自动释放，不需要手动清理
    }

private:
    std::unique_ptr<std::vector<int>> resource1_;
    std::unique_ptr<std::string> resource2_;
};

void test_multi_resource() {
    std::cout << "\n=== Test 7: Multiple Resources ===" << std::endl;

    {
        MultiResource res;
        // 使用资源...
    }  // 所有资源自动释放

    std::cout << "All resources cleaned up!" << std::endl;
}

// ============ 主函数 ============

int main() {
    std::cout << "RAII Examples\n" << std::endl;
    std::cout << "=================================================\n" << std::endl;

    try {
        test_file_raii();
        test_lock_raii();
        test_timer_raii();
        test_unique_ptr();
        test_exception_safety();
        test_manual_vs_raii();
        test_multi_resource();

        std::cout << "\n=================================================\n" << std::endl;
        std::cout << "All tests passed! ✅" << std::endl;
        std::cout << "\n关键收获：" << std::endl;
        std::cout << "1. RAII 自动管理资源，不会泄漏" << std::endl;
        std::cout << "2. 异常安全：即使抛异常也会清理" << std::endl;
        std::cout << "3. 代码简洁：不需要手动清理" << std::endl;
        std::cout << "4. 编译器保证：离开作用域一定调用析构" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
