// RAII 实践：管理底层 FILE* 资源
// 编译：g++ -std=c++17 07_raii_file_manager.cpp -o file_mgr
//
// 目的：演示如何用 RAII 管理需要手动释放的底层资源（FILE*）
// 对比：标准库的 fstream 已经是 RAII 的，不需要再包装

#include <iostream>
#include <cstdio>    // FILE*, fopen, fclose
#include <stdexcept>
#include <string>
#include <vector>

// RAII 包装器：管理 FILE* 资源
class FileHandle {
    FILE* file_;
    std::string path_;

public:
    // 1. 构造时获取资源
    FileHandle(const char* path, const char* mode) : path_(path) {
        file_ = fopen(path, mode);
        if (!file_) {
            throw std::runtime_error("Failed to open: " + path_);
        }
        std::cout << "✓ Opened: " << path_ << " (mode: " << mode << ")\n";
    }

    // 2. 析构时释放资源
    ~FileHandle() {
        if (file_) {
            fclose(file_);
            std::cout << "✓ Closed: " << path_ << "\n";
        }
    }

    // 3. 禁止拷贝（防止重复释放）
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // 4. 移动语义（转移所有权）
    FileHandle(FileHandle&& other) noexcept
        : file_(other.file_), path_(std::move(other.path_)) {
        other.file_ = nullptr;
    }

    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            if (file_) fclose(file_);
            file_ = other.file_;
            path_ = std::move(other.path_);
            other.file_ = nullptr;
        }
        return *this;
    }

    // 5. 提供操作接口
    void write(const std::string& content) {
        if (file_) {
            fputs(content.c_str(), file_);
        }
    }

    std::string read_line() {
        char buffer[256];
        if (file_ && fgets(buffer, sizeof(buffer), file_)) {
            return std::string(buffer);
        }
        return "";
    }

    FILE* get() const { return file_; }
};

// 演示：管理多个文件资源
void demo_multiple_files() {
    std::cout << "\n=== 演示：管理多个文件 ===\n";

    std::vector<FileHandle> files;

    // 打开多个文件（自动管理）
    files.push_back(FileHandle("file1.txt", "w"));
    files.push_back(FileHandle("file2.txt", "w"));
    files.push_back(FileHandle("file3.txt", "w"));

    // 写入数据
    for (size_t i = 0; i < files.size(); ++i) {
        files[i].write("Data from file " + std::to_string(i + 1) + "\n");
    }

    // 离开作用域，所有文件自动关闭（RAII）
    std::cout << "离开作用域...\n";
}

// 演示：异常安全
void demo_exception_safety() {
    std::cout << "\n=== 演示：异常安全 ===\n";

    try {
        FileHandle f1("temp1.txt", "w");
        f1.write("Some data\n");

        // 模拟异常
        throw std::runtime_error("模拟错误");

        // 这行不会执行
        FileHandle f2("temp2.txt", "w");

    } catch (const std::exception& e) {
        std::cout << "捕获异常: " << e.what() << "\n";
        std::cout << "注意：f1 仍然被正确关闭（RAII 保证）\n";
    }
}

// 演示：移动语义
FileHandle create_file(const char* path) {
    std::cout << "\n=== 演示：移动语义（返回值） ===\n";
    FileHandle f(path, "w");
    f.write("Created by factory function\n");
    return f;  // 自动移动（RVO 或移动构造）
}

int main() {
    std::cout << "RAII 底层资源管理演示\n";
    std::cout << "========================\n";

    try {
        // 1. 基本使用
        {
            std::cout << "\n=== 演示：基本使用 ===\n";
            FileHandle f("test.txt", "w");
            f.write("Hello RAII\n");
            f.write("Automatic resource management\n");
        }  // f 析构，文件自动关闭

        // 2. 管理多个文件
        demo_multiple_files();

        // 3. 异常安全
        demo_exception_safety();

        // 4. 移动语义
        auto f = create_file("moved.txt");
        std::cout << "文件已移动到 main\n";

        std::cout << "\n=== 所有演示完成 ===\n";

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
    }

    std::cout << "\n程序结束，所有资源已释放\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. RAII 五要素：
 *    - 构造时获取资源
 *    - 析构时释放资源
 *    - 禁止拷贝（= delete）
 *    - 实现移动（转移所有权）
 *    - 提供访问接口
 *
 * 2. 为什么要自己写？
 *    - FILE* 是 C 风格资源，需要手动 fclose
 *    - 标准库的 fstream 已经是 RAII 的，日常用它
 *    - 这个例子展示如何包装底层资源
 *
 * 3. 实际项目中：
 *    - 文件操作 → 用 std::fstream（已经 RAII）
 *    - 套接字、数据库连接等 → 用 RAII 包装
 *    - 遵循 Rule of 0：优先用标准库
 */
