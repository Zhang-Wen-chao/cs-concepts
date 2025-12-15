// RAII 文件管理器（最小示例）
// 编译：g++ -std=c++17 07_raii_file_manager.cpp -o file_mgr

#include <iostream>
#include <fstream>
#include <string>

class FileManager {
    std::fstream file_;
    std::string path_;
public:
    FileManager(const std::string& path, bool write = false) : path_(path) {
        auto mode = write ? (std::ios::out | std::ios::trunc) : std::ios::in;
        file_.open(path, mode);
        if (!file_) throw std::runtime_error("Cannot open: " + path);
        std::cout << "Opened: " << path << "\n";
    }

    ~FileManager() {
        if (file_.is_open()) {
            file_.close();
            std::cout << "Closed: " << path_ << "\n";
        }
    }

    void write(const std::string& content) {
        file_ << content;
    }

    std::string read() {
        std::string content, line;
        while (std::getline(file_, line)) {
            content += line + "\n";
        }
        return content;
    }
};

int main() {
    try {
        // 写文件
        {
            FileManager fm("test.txt", true);
            fm.write("Hello RAII\n");
        }  // 自动关闭

        // 读文件
        {
            FileManager fm("test.txt");
            std::cout << fm.read();
        }  // 自动关闭

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
