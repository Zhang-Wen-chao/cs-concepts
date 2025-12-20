// 网络 I/O 实践：简单的 Echo 服务器
// 编译：g++ -std=c++17 01_echo_server.cpp -o echo_server
// 运行：./echo_server
// 测试：telnet localhost 8080 或 nc localhost 8080
//
// 目的：演示基本的 TCP socket 编程

#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

// RAII Socket 管理
class Socket {
    int fd_;
public:
    Socket(int fd) : fd_(fd) {}
    ~Socket() {
        if (fd_ >= 0) {
            close(fd_);
            std::cout << "Socket " << fd_ << " 已关闭\n";
        }
    }

    int get() const { return fd_; }
    bool valid() const { return fd_ >= 0; }

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
};

// 创建服务器 socket
int create_server_socket(int port) {
    // 1. 创建 socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }

    // 2. 设置 SO_REUSEADDR（允许立即重用端口）
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        close(server_fd);
        return -1;
    }

    // 3. 绑定地址
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;  // 0.0.0.0（监听所有接口）
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }

    // 4. 监听
    if (listen(server_fd, 10) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    std::cout << "服务器启动在端口 " << port << "\n";
    return server_fd;
}

// 处理客户端连接
void handle_client(int client_fd) {
    char buffer[1024];

    while (true) {
        memset(buffer, 0, sizeof(buffer));

        // 读取数据
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);

        if (bytes_read < 0) {
            perror("read");
            break;
        }

        if (bytes_read == 0) {
            std::cout << "客户端断开连接\n";
            break;
        }

        std::cout << "收到: " << buffer;

        // 回显数据
        ssize_t bytes_written = write(client_fd, buffer, bytes_read);
        if (bytes_written < 0) {
            perror("write");
            break;
        }
    }
}

int main() {
    const int PORT = 8080;

    // 创建服务器 socket
    int server_fd = create_server_socket(PORT);
    if (server_fd < 0) {
        return 1;
    }

    Socket server_socket(server_fd);

    std::cout << "等待连接...\n";
    std::cout << "测试命令: telnet localhost " << PORT << "\n";
    std::cout << "或者: nc localhost " << PORT << "\n";

    while (true) {
        // 接受连接
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        std::cout << "\n新连接: socket fd = " << client_fd << "\n";

        {
            Socket client_socket(client_fd);
            handle_client(client_fd);
        }  // client_socket 析构，自动关闭连接

        std::cout << "连接已关闭\n";
    }

    return 0;
}

/*
 * 关键要点：
 *
 * 1. TCP 服务器流程：
 *    socket() → bind() → listen() → accept() → read/write → close()
 *
 * 2. socket() 参数：
 *    AF_INET      - IPv4
 *    SOCK_STREAM  - TCP
 *    0            - 默认协议
 *
 * 3. bind() 地址：
 *    INADDR_ANY   - 0.0.0.0（监听所有接口）
 *    htons()      - 主机字节序转网络字节序
 *
 * 4. listen() 参数：
 *    backlog      - 等待队列大小
 *
 * 5. accept()：
 *    阻塞，直到有新连接
 *    返回新的 socket fd
 *
 * 6. SO_REUSEADDR：
 *    允许立即重用端口（避免 TIME_WAIT）
 *
 * 7. RAII 管理 socket：
 *    自动 close()，避免泄漏
 *
 * 8. 错误处理：
 *    检查所有返回值
 *    用 perror() 打印错误
 */
