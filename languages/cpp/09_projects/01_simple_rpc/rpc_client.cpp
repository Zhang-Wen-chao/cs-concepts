#include "rpc_client.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace simple_rpc {

RpcClient::RpcClient(const std::string& host, int port)
    : host_(host), port_(port), sock_fd_(-1) {}

RpcClient::~RpcClient() {
    if (sock_fd_ >= 0) {
        close(sock_fd_);
    }
}

bool RpcClient::connect() {
    // 创建 socket
    sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
        perror("socket");
        return false;
    }

    // 设置服务器地址
    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr) <= 0) {
        std::cerr << "无效的地址: " << host_ << "\n";
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    // 连接服务器
    if (::connect(sock_fd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect");
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    std::cout << "连接到服务器 " << host_ << ":" << port_ << "\n";
    return true;
}

int RpcClient::call(const std::string& function_name, int arg1, int arg2) {
    std::cout << "[调试] call() 开始: " << function_name << "(" << arg1 << ", " << arg2 << ")\n" << std::flush;

    if (sock_fd_ < 0) {
        std::cout << "[调试] socket 未连接，尝试连接...\n" << std::flush;
        if (!connect()) {
            throw std::runtime_error("连接服务器失败");
        }
    }

    uint32_t function_id = hash_function_name(function_name);
    std::cout << "[调试] 函数ID: " << function_id << "\n" << std::flush;

    std::cout << "[调试] 发送请求...\n" << std::flush;
    if (!send_request(function_id, arg1, arg2)) {
        throw std::runtime_error("发送请求失败");
    }

    std::cout << "[调试] 等待响应...\n" << std::flush;
    return receive_response();
}

bool RpcClient::send_request(uint32_t function_id, int arg1, int arg2) {
    // 1. 序列化参数
    Serializer serializer;
    serializer.write_int(arg1);
    serializer.write_int(arg2);

    // 2. 构造消息头
    MessageHeader header;
    header.length = serializer.size();
    header.type = static_cast<uint32_t>(MessageType::REQUEST);
    header.function_id = function_id;
    header.to_network_order();

    // 3. 发送消息头
    ssize_t n = write(sock_fd_, &header, sizeof(header));
    if (n != sizeof(header)) {
        return false;
    }

    // 4. 发送消息体
    n = write(sock_fd_, serializer.data().data(), serializer.size());
    if (n != static_cast<ssize_t>(serializer.size())) {
        return false;
    }

    return true;
}

int RpcClient::receive_response() {
    // 1. 读取消息头
    MessageHeader header;
    ssize_t n = read(sock_fd_, &header, sizeof(header));
    if (n != sizeof(header)) {
        throw std::runtime_error("读取响应头失败");
    }
    header.from_network_order();

    // 2. 检查状态
    StatusCode status = static_cast<StatusCode>(header.function_id);
    if (status != StatusCode::OK) {
        throw std::runtime_error("服务器返回错误: " + std::to_string(header.function_id));
    }

    // 3. 读取消息体
    std::vector<char> body(header.length);
    n = read(sock_fd_, body.data(), header.length);
    if (n != static_cast<ssize_t>(header.length)) {
        throw std::runtime_error("读取响应体失败");
    }

    // 4. 反序列化结果
    Serializer deserializer;
    deserializer.set_data(body);
    return deserializer.read_int();
}

} // namespace simple_rpc
