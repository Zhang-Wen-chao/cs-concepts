#include "rpc_server.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

namespace simple_rpc {

RpcServer::RpcServer(int port) : port_(port), server_fd_(-1), running_(false) {}

RpcServer::~RpcServer() {
    stop();
}

void RpcServer::register_function(const std::string& name, std::function<int(int, int)> func) {
    uint32_t function_id = hash_function_name(name);
    functions_[function_id] = func;
    std::cout << "注册函数: " << name << " (ID: " << function_id << ")\n";
}

bool RpcServer::create_server_socket() {
    // 创建 socket
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        perror("socket");
        return false;
    }

    // 设置 SO_REUSEADDR
    int opt = 1;
    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd_);
        return false;
    }

    // 绑定地址
    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind");
        close(server_fd_);
        return false;
    }

    // 监听
    if (listen(server_fd_, 10) < 0) {
        perror("listen");
        close(server_fd_);
        return false;
    }

    return true;
}

void RpcServer::start() {
    if (!create_server_socket()) {
        throw std::runtime_error("创建服务器 socket 失败");
    }

    running_ = true;
    std::cout << "RPC 服务器启动在端口 " << port_ << "\n";
    std::cout << "等待连接...\n";

    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (running_) {
                perror("accept");
            }
            continue;
        }

        std::cout << "客户端连接\n";

        // 简化：单线程处理（实际应用应该用线程池）
        handle_client(client_fd);

        close(client_fd);
        std::cout << "客户端断开\n";
    }
}

void RpcServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }
}

void RpcServer::handle_client(int client_fd) {
    while (true) {
        try {
            handle_request(client_fd);
        } catch (const std::exception& e) {
            std::cerr << "处理请求错误: " << e.what() << "\n";
            break;
        }
    }
}

void RpcServer::handle_request(int client_fd) {
    // 1. 读取消息头
    MessageHeader header;
    ssize_t n = read(client_fd, &header, sizeof(header));
    if (n <= 0) {
        throw std::runtime_error("读取消息头失败");
    }
    header.from_network_order();

    // 2. 读取消息体
    std::vector<char> body(header.length);
    n = read(client_fd, body.data(), header.length);
    if (n != static_cast<ssize_t>(header.length)) {
        throw std::runtime_error("读取消息体失败");
    }

    // 3. 反序列化参数
    Serializer deserializer;
    deserializer.set_data(body);
    int arg1 = deserializer.read_int();
    int arg2 = deserializer.read_int();

    std::cout << "收到请求: 函数ID " << header.function_id
              << "(" << arg1 << ", " << arg2 << ")\n";

    // 4. 查找并调用函数
    StatusCode status = StatusCode::OK;
    int result = 0;

    auto it = functions_.find(header.function_id);
    if (it == functions_.end()) {
        status = StatusCode::FUNCTION_NOT_FOUND;
        std::cerr << "函数未找到: " << header.function_id << "\n";
    } else {
        result = it->second(arg1, arg2);
        std::cout << "返回结果: " << result << "\n";
    }

    // 5. 序列化返回值
    Serializer serializer;
    serializer.write_int(result);

    // 6. 发送响应
    MessageHeader response_header;
    response_header.length = serializer.size();
    response_header.type = static_cast<uint32_t>(MessageType::RESPONSE);
    response_header.function_id = static_cast<uint32_t>(status);
    response_header.to_network_order();

    write(client_fd, &response_header, sizeof(response_header));
    write(client_fd, serializer.data().data(), serializer.size());
}

} // namespace simple_rpc
