// RAII 核心示例
// 编译：g++ -std=c++17 01_raii_examples.cpp -o raii

#include <iostream>
#include <fstream>
#include <mutex>

// 示例1：文件 RAII
class FileHandle {
    FILE* file_;
public:
    FileHandle(const char* name) : file_(fopen(name, "r")) {
        if (!file_) throw std::runtime_error("Cannot open file");
    }
    ~FileHandle() { if (file_) fclose(file_); }
    FILE* get() const { return file_; }
};

// 示例2：锁 RAII
std::mutex mtx;
int shared_data = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mtx);  // 自动加锁
    ++shared_data;
}  // 自动解锁

// 示例3：计时器 RAII
class Timer {
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    Timer(const char* name) : name_(name), start_(std::chrono::steady_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << name_ << " took " << ms << "ms\n";
    }
};

int main() {
    // 测试文件 RAII
    try {
        FileHandle f("test.txt");
        std::cout << "File opened\n";
    } catch (...) {
        std::cout << "File error\n";
    }

    // 测试锁 RAII
    increment();

    // 测试计时器
    {
        Timer t("Operation");
        // 耗时操作...
    }  // 自动打印耗时
}
