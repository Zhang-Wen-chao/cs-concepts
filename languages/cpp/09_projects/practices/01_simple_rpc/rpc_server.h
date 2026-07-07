#pragma once

#include "rpc_protocol.h"
#include <functional>
#include <unordered_map>
#include <memory>
#include <iostream>

namespace simple_rpc {

class RpcServer {
    int port_;
    int server_fd_;
    std::unordered_map<uint32_t, std::function<int(int, int)>> functions_;
    bool running_;

public:
    explicit RpcServer(int port);
    ~RpcServer();

    // 注册函数（目前只支持 int(int, int) 类型）
    void register_function(const std::string& name, std::function<int(int, int)> func);

    // 启动服务器
    void start();

    // 停止服务器
    void stop();

private:
    // 创建服务器 socket
    bool create_server_socket();

    // 处理客户端连接
    void handle_client(int client_fd);

    // 处理单个请求
    void handle_request(int client_fd);
};

} // namespace simple_rpc
