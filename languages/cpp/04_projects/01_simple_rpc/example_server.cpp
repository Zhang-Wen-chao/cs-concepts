// RPC 服务器示例
// 编译：g++ -std=c++17 rpc_server.cpp example_server.cpp -o server
// 运行：./server

#include "rpc_server.h"
#include <iostream>
#include <csignal>

using namespace simple_rpc;

// 全局服务器（用于信号处理）
RpcServer* g_server = nullptr;

void signal_handler(int signal) {
    std::cout << "\n收到信号 " << signal << "，停止服务器...\n";
    if (g_server) {
        g_server->stop();
    }
    exit(0);
}

// 业务函数
int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    return a * b;
}

int divide(int a, int b) {
    if (b == 0) {
        std::cerr << "除数为 0\n";
        return 0;
    }
    return a / b;
}

int main() {
    std::cout << "=== Simple RPC 服务器 ===\n\n";

    // 设置信号处理
    std::signal(SIGINT, signal_handler);

    try {
        RpcServer server(9090);  // 改为 9090 端口
        g_server = &server;

        // 注册服务
        server.register_function("add", add);
        server.register_function("subtract", subtract);
        server.register_function("multiply", multiply);
        server.register_function("divide", divide);

        std::cout << "\n服务列表：\n";
        std::cout << "  - add(a, b): 加法\n";
        std::cout << "  - subtract(a, b): 减法\n";
        std::cout << "  - multiply(a, b): 乘法\n";
        std::cout << "  - divide(a, b): 除法\n";
        std::cout << "\n";

        // 启动服务器（阻塞）
        server.start();

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

/*
 * 运行说明：
 *
 * 1. 编译：
 *    g++ -std=c++17 rpc_server.cpp example_server.cpp -o server
 *
 * 2. 运行：
 *    ./server
 *
 * 3. 测试：
 *    在另一个终端运行客户端：
 *    ./client
 *
 * 4. 停止：
 *    Ctrl+C
 */
