// RPC 客户端示例
// 编译：g++ -std=c++17 rpc_client.cpp example_client.cpp -o client
// 运行：./client

#include "rpc_client.h"
#include <iostream>

using namespace simple_rpc;

int main() {
    std::cout << "=== Simple RPC 客户端 ===\n\n";

    try {
        RpcClient client("127.0.0.1", 9090);  // 改为 9090 端口

        // 测试加法
        std::cout << "测试加法：\n";
        int sum = client.call("add", 10, 20);
        std::cout << "  10 + 20 = " << sum << "\n\n";

        // 测试减法
        std::cout << "测试减法：\n";
        int diff = client.call("subtract", 100, 25);
        std::cout << "  100 - 25 = " << diff << "\n\n";

        // 测试乘法
        std::cout << "测试乘法：\n";
        int product = client.call("multiply", 6, 7);
        std::cout << "  6 * 7 = " << product << "\n\n";

        // 测试除法
        std::cout << "测试除法：\n";
        int quotient = client.call("divide", 100, 5);
        std::cout << "  100 / 5 = " << quotient << "\n\n";

        // 测试多次调用
        std::cout << "测试多次调用：\n";
        for (int i = 1; i <= 5; ++i) {
            int result = client.call("add", i, i);
            std::cout << "  " << i << " + " << i << " = " << result << "\n";
        }

        std::cout << "\n所有测试完成！\n";

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

/*
 * 运行说明：
 *
 * 1. 确保服务器已启动：
 *    ./server
 *
 * 2. 编译客户端：
 *    g++ -std=c++17 rpc_client.cpp example_client.cpp -o client
 *
 * 3. 运行客户端：
 *    ./client
 *
 * 4. 预期输出：
 *    === Simple RPC 客户端 ===
 *
 *    连接到服务器 127.0.0.1:8080
 *    测试加法：
 *      10 + 20 = 30
 *
 *    测试减法：
 *      100 - 25 = 75
 *
 *    测试乘法：
 *      6 * 7 = 42
 *
 *    测试除法：
 *      100 / 5 = 20
 *
 *    测试多次调用：
 *      1 + 1 = 2
 *      2 + 2 = 4
 *      3 + 3 = 6
 *      4 + 4 = 8
 *      5 + 5 = 10
 *
 *    所有测试完成！
 */
