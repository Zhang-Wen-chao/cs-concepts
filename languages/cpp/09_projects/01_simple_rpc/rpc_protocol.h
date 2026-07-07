#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <arpa/inet.h>

namespace simple_rpc {

// 消息类型
enum class MessageType : uint32_t {
    REQUEST = 1,
    RESPONSE = 2
};

// 状态码
enum class StatusCode : uint32_t {
    OK = 0,
    ERROR = 1,
    FUNCTION_NOT_FOUND = 2,
    INVALID_PARAMS = 3
};

// 消息头（固定 12 字节）
struct MessageHeader {
    uint32_t length;        // 消息总长度（不含 header）
    uint32_t type;          // 消息类型
    uint32_t function_id;   // 函数ID 或 状态码

    // 序列化为网络字节序
    void to_network_order() {
        length = htonl(length);
        type = htonl(type);
        function_id = htonl(function_id);
    }

    // 从网络字节序转换
    void from_network_order() {
        length = ntohl(length);
        type = ntohl(type);
        function_id = ntohl(function_id);
    }
};

// 简单的序列化器（支持 int、string）
class Serializer {
    std::vector<char> buffer_;
    size_t read_pos_ = 0;

public:
    // 写入 int
    void write_int(int value) {
        uint32_t net_value = htonl(static_cast<uint32_t>(value));
        const char* p = reinterpret_cast<const char*>(&net_value);
        buffer_.insert(buffer_.end(), p, p + sizeof(net_value));
    }

    // 写入 string
    void write_string(const std::string& str) {
        write_int(static_cast<int>(str.size()));  // 先写长度
        buffer_.insert(buffer_.end(), str.begin(), str.end());
    }

    // 读取 int
    int read_int() {
        if (read_pos_ + sizeof(uint32_t) > buffer_.size()) {
            throw std::runtime_error("读取越界");
        }
        uint32_t net_value;
        std::memcpy(&net_value, &buffer_[read_pos_], sizeof(net_value));
        read_pos_ += sizeof(net_value);
        return static_cast<int>(ntohl(net_value));
    }

    // 读取 string
    std::string read_string() {
        int len = read_int();
        if (read_pos_ + len > buffer_.size()) {
            throw std::runtime_error("读取越界");
        }
        std::string str(buffer_.begin() + read_pos_, buffer_.begin() + read_pos_ + len);
        read_pos_ += len;
        return str;
    }

    // 获取数据
    const std::vector<char>& data() const { return buffer_; }

    // 设置数据（用于反序列化）
    void set_data(const std::vector<char>& data) {
        buffer_ = data;
        read_pos_ = 0;
    }

    // 大小
    size_t size() const { return buffer_.size(); }
};

// 函数名 → 函数ID 的映射（简化：用哈希）
inline uint32_t hash_function_name(const std::string& name) {
    uint32_t hash = 0;
    for (char c : name) {
        hash = hash * 31 + c;
    }
    return hash;
}

} // namespace simple_rpc
