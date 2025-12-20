// 网络 I/O 实践：简单的 Echo 客户端
// 编译：g++ -std=c++17 01_echo_client.cpp -o echo_client
// 运行：./echo_client
//
// 目的：演示基本的 TCP 客户端编程

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

// RAII Socket 管理
class Socket {
    int fd_;
public:
    Socket(int fd) : fd_(fd) {}
    ~Socket() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    int get() const { return fd_; }
    bool valid() const { return fd_ >= 0; }

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
};

int main() {
    const char* SERVER_IP = "127.0.0.1";
    const int SERVER_PORT = 8080;

    // 1. 创建 socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }

    Socket client_socket(sock);

    // 2. 设置服务器地址
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SERVER_PORT);

    // 转换 IP 地址
    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        std::cerr << "无效的地址\n";
        return 1;
    }

    // 3. 连接服务器
    std::cout << "连接到 " << SERVER_IP << ":" << SERVER_PORT << "...\n";

    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect");
        return 1;
    }

    std::cout << "已连接！输入消息（输入 'quit' 退出）:\n";

    // 4. 读写数据
    while (true) {
        // 读取用户输入
        std::string message;
        std::cout << "> ";
        std::getline(std::cin, message);

        if (message == "quit") {
            break;
        }

        // 发送数据
        ssize_t bytes_sent = send(sock, message.c_str(), message.length(), 0);
        if (bytes_sent < 0) {
            perror("send");
            break;
        }

        // 接收回显
        char buffer[1024] = {0};
        ssize_t bytes_received = read(sock, buffer, sizeof(buffer) - 1);

        if (bytes_received < 0) {
            perror("read");
            break;
        }

        if (bytes_received == 0) {
            std::cout << "服务器断开连接\n";
            break;
        }

        std::cout << "收到回显: " << buffer << "\n";
    }

    std::cout << "断开连接\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. TCP 客户端流程：
 *    socket() → connect() → read/write → close()
 *
 * 2. inet_pton()：
 *    将 IP 地址字符串转换为网络字节序
 *    AF_INET：IPv4
 *
 * 3. connect()：
 *    连接到服务器
 *    阻塞，直到连接成功或失败
 *
 * 4. send() vs write()：
 *    send() 可以设置标志（如 MSG_NOSIGNAL）
 *    write() 更通用
 *
 * 5. RAII 管理 socket：
 *    自动 close()
 *
 * 6. 错误处理：
 *    检查所有返回值
 */
