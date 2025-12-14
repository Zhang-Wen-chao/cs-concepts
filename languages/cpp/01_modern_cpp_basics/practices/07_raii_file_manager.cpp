/**
 * 实践项目 1：RAII 风格的文件管理类
 *
 * 编译：g++ -std=c++17 07_raii_file_manager.cpp -o raii_file_manager
 * 运行：./raii_file_manager
 */

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <ctime>

namespace fs = std::filesystem;

class ScopedFile {
public:
    enum class Mode { kRead, kWrite, kAppend, kReadWrite };

    ScopedFile(const fs::path& path, Mode mode) : path_(path), mode_(mode) {
        if (path_.empty()) {
            throw std::invalid_argument("File path cannot be empty");
        }
        Open();
    }

    ~ScopedFile() { Close(); }

    ScopedFile(const ScopedFile&) = delete;
    ScopedFile& operator=(const ScopedFile&) = delete;

    ScopedFile(ScopedFile&& other) noexcept : file_(other.file_), path_(std::move(other.path_)), mode_(other.mode_) {
        other.file_ = nullptr;
    }

    ScopedFile& operator=(ScopedFile&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        Close();
        file_ = other.file_;
        path_ = std::move(other.path_);
        mode_ = other.mode_;
        other.file_ = nullptr;
        other.path_.clear();
        other.mode_ = Mode::kRead;
        return *this;
    }

    void Write(std::string_view data) {
        EnsureWritable();
        if (data.empty()) {
            return;
        }
        const auto written = std::fwrite(data.data(), 1, data.size(), file_);
        if (written != data.size()) {
            throw std::runtime_error("Failed to write to file: " + path_.string());
        }
        std::fflush(file_);
    }

    void WriteLine(std::string_view line) {
        Write(line);
        Write("\n");
    }

    void AppendLine(std::string_view line) {
        MoveToEnd();
        WriteLine(line);
    }

    std::string ReadAll() {
        EnsureReadable();
        std::fflush(file_);
        if (std::fseek(file_, 0, SEEK_END) != 0) {
            throw std::runtime_error("Failed to seek in file: " + path_.string());
        }
        const auto file_size = std::ftell(file_);
        if (file_size < 0) {
            throw std::runtime_error("Failed to learn file size: " + path_.string());
        }
        std::string contents(static_cast<size_t>(file_size), '\0');
        std::rewind(file_);
        if (!contents.empty()) {
            const auto read_bytes = std::fread(contents.data(), 1, contents.size(), file_);
            contents.resize(read_bytes);
        }
        return contents;
    }

    std::vector<std::string> ReadLines() {
        std::vector<std::string> result;
        std::istringstream stream(ReadAll());
        std::string line;
        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            result.push_back(line);
        }
        return result;
    }

    void Reset(Mode new_mode) {
        Close();
        mode_ = new_mode;
        Open();
    }

    std::uintmax_t FileSizeOnDisk() const {
        std::error_code ec;
        return fs::file_size(path_, ec);
    }

    const fs::path& path() const { return path_; }

private:
    struct ModeTraits {
        const char* fopen_mode;
        bool readable;
        bool writable;
    };

    static ModeTraits Traits(Mode mode) {
        switch (mode) {
            case Mode::kRead:
                return {"rb", true, false};
            case Mode::kWrite:
                return {"wb", false, true};
            case Mode::kAppend:
                return {"ab", false, true};
            case Mode::kReadWrite:
                return {"w+b", true, true};
            default:
                throw std::logic_error("Unknown file mode");
        }
    }

    void EnsureReadable() const {
        if (!file_) {
            throw std::runtime_error("File handle is not open");
        }
        if (!Traits(mode_).readable) {
            throw std::logic_error("File was not opened for reading: " + path_.string());
        }
    }

    void EnsureWritable() const {
        if (!file_) {
            throw std::runtime_error("File handle is not open");
        }
        if (!Traits(mode_).writable) {
            throw std::logic_error("File was not opened for writing: " + path_.string());
        }
    }

    void MoveToEnd() {
        EnsureWritable();
        if (std::fseek(file_, 0, SEEK_END) != 0) {
            throw std::runtime_error("Failed to seek to end of file: " + path_.string());
        }
    }

    void Open() {
        const auto traits = Traits(mode_);
        file_ = std::fopen(path_.string().c_str(), traits.fopen_mode);
        if (!file_) {
            throw std::runtime_error("Cannot open file '" + path_.string() + "': " + std::strerror(errno));
        }
    }

    void Close() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    FILE* file_ = nullptr;
    fs::path path_;
    Mode mode_ = Mode::kRead;
};

std::string CurrentTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local{};
#if defined(_WIN32)
    localtime_s(&tm_local, &tt);
#else
    localtime_r(&tt, &tm_local);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_local, "%F %T");
    return oss.str();
}

void RunRaiiFileManagerDemo() {
    std::cout << "\n=== RAII 文件管理器示例 ===" << std::endl;
    const fs::path file_path("raii_file_demo.log");
    std::error_code ec;
    fs::remove(file_path, ec);

    try {
        {
            ScopedFile writer(file_path, ScopedFile::Mode::kWrite);
            writer.WriteLine("log started at " + CurrentTimestamp());
            writer.WriteLine("collecting startup data...");
            writer.WriteLine("system ready");
            std::cout << "初始写入完成, 大小约 " << writer.FileSizeOnDisk() << " bytes" << std::endl;
        }

        {
            ScopedFile appender(file_path, ScopedFile::Mode::kAppend);
            appender.AppendLine("first event recorded");
            appender.AppendLine("second event recorded");
            std::cout << "追加两条日志完成" << std::endl;
        }

        {
            ScopedFile reader(file_path, ScopedFile::Mode::kRead);
            auto lines = reader.ReadLines();
            std::cout << "读取 " << lines.size() << " 行日志:" << std::endl;
            for (const auto& line : lines) {
                std::cout << "  - " << line << std::endl;
            }
        }

        try {
            ScopedFile read_only(file_path, ScopedFile::Mode::kRead);
            read_only.AppendLine("this should fail");
        } catch (const std::exception& e) {
            std::cout << "尝试在只读模式写入 -> 捕获到异常: " << e.what() << std::endl;
        }

        std::cout << "最终文件大小: " << fs::file_size(file_path) << " bytes" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "文件管理器示例失败: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   现代 C++ 实践 1：RAII 文件管理器" << std::endl;
    std::cout << "============================================" << std::endl;

    RunRaiiFileManagerDemo();

    std::cout << "\n示例执行完毕！" << std::endl;
    return 0;
}
