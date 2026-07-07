# 网络 I/O

> 理解 socket 编程和网络通信基础

## 核心概念

**socket = 网络通信的端点**

```
客户端                     服务器
 ┌────┐                   ┌────┐
 │App │                   │App │
 └─┬──┘                   └─┬──┘
   │                        │
 socket                   socket
   │                        │
   └────── 网络连接 ────────┘
```

## TCP vs UDP

**TCP（可靠）**：
- 面向连接
- 可靠传输（保证顺序、不丢失）
- 三次握手
- 适用：HTTP、文件传输、数据库

**UDP（快速）**：
- 无连接
- 不可靠（可能丢失、乱序）
- 无握手
- 适用：视频流、游戏、DNS

## TCP 服务器基本流程

```cpp
1. socket()   - 创建 socket
2. bind()     - 绑定地址和端口
3. listen()   - 监听连接
4. accept()   - 接受连接（阻塞）
5. read/write - 读写数据
6. close()    - 关闭连接
```

## TCP 客户端基本流程

```cpp
1. socket()   - 创建 socket
2. connect()  - 连接服务器
3. read/write - 读写数据
4. close()    - 关闭连接
```

## 简单 TCP 服务器

```cpp
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

// 创建 socket
int server_fd = socket(AF_INET, SOCK_STREAM, 0);

// 设置地址
struct sockaddr_in address;
address.sin_family = AF_INET;
address.sin_addr.s_addr = INADDR_ANY;  // 0.0.0.0
address.sin_port = htons(8080);        // 端口 8080

// 绑定
bind(server_fd, (struct sockaddr*)&address, sizeof(address));

// 监听
listen(server_fd, 3);  // 最多 3 个等待连接

// 接受连接
int client_fd = accept(server_fd, nullptr, nullptr);

// 读写数据
char buffer[1024] = {0};
read(client_fd, buffer, 1024);
write(client_fd, "Hello", 5);

// 关闭
close(client_fd);
close(server_fd);
```

## 简单 TCP 客户端

```cpp
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// 创建 socket
int sock = socket(AF_INET, SOCK_STREAM, 0);

// 服务器地址
struct sockaddr_in serv_addr;
serv_addr.sin_family = AF_INET;
serv_addr.sin_port = htons(8080);
inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);

// 连接
connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

// 发送数据
send(sock, "Hello", 5, 0);

// 接收数据
char buffer[1024] = {0};
read(sock, buffer, 1024);

// 关闭
close(sock);
```

## 非阻塞 I/O

**阻塞 I/O（默认）**：
```cpp
// accept() 会阻塞，直到有连接
int client_fd = accept(server_fd, nullptr, nullptr);

// read() 会阻塞，直到有数据
read(client_fd, buffer, 1024);
```

**非阻塞 I/O**：
```cpp
#include <fcntl.h>

// 设置非阻塞
int flags = fcntl(server_fd, F_GETFL, 0);
fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

// accept() 立即返回
int client_fd = accept(server_fd, nullptr, nullptr);
if (client_fd < 0) {
    // EAGAIN 或 EWOULDBLOCK：没有连接
}
```

## I/O 多路复用：select

**问题**：如何同时处理多个连接？

**select**：监听多个 socket
```cpp
fd_set read_fds;
FD_ZERO(&read_fds);
FD_SET(server_fd, &read_fds);

// 等待事件（阻塞）
int activity = select(max_fd + 1, &read_fds, nullptr, nullptr, nullptr);

// 检查哪个 socket 有事件
if (FD_ISSET(server_fd, &read_fds)) {
    // server_fd 有新连接
}
```

## I/O 多路复用：epoll（Linux）

**epoll 更高效**（比 select 快）

```cpp
#include <sys/epoll.h>

// 创建 epoll 实例
int epoll_fd = epoll_create1(0);

// 添加 socket 到 epoll
struct epoll_event event;
event.events = EPOLLIN;  // 监听读事件
event.data.fd = server_fd;
epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &event);

// 等待事件
struct epoll_event events[10];
int n = epoll_wait(epoll_fd, events, 10, -1);

// 处理事件
for (int i = 0; i < n; ++i) {
    if (events[i].data.fd == server_fd) {
        // 新连接
        int client_fd = accept(server_fd, nullptr, nullptr);
        // 添加到 epoll
        event.data.fd = client_fd;
        epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &event);
    } else {
        // 数据到达
        int fd = events[i].data.fd;
        read(fd, buffer, 1024);
    }
}
```

## 简单 HTTP 服务器

```cpp
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

void handle_client(int client_fd) {
    char buffer[1024] = {0};
    read(client_fd, buffer, 1024);

    // HTTP 响应
    const char* response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html\r\n"
        "\r\n"
        "<html><body><h1>Hello World</h1></body></html>";

    write(client_fd, response, strlen(response));
    close(client_fd);
}

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    bind(server_fd, (struct sockaddr*)&address, sizeof(address));
    listen(server_fd, 3);

    std::cout << "服务器启动在 http://localhost:8080\n";

    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        handle_client(client_fd);
    }

    close(server_fd);
    return 0;
}
```

## 常见陷阱

### 陷阱 1：忘记 close

```cpp
// ❌ 忘记关闭，文件描述符泄漏
int sock = socket(AF_INET, SOCK_STREAM, 0);
// ... 使用 ...
// 忘记 close(sock)

// ✅ RAII 管理
class Socket {
    int fd_;
public:
    Socket(int fd) : fd_(fd) {}
    ~Socket() { if (fd_ >= 0) close(fd_); }
    // ... 禁止拷贝，实现移动 ...
};
```

### 陷阱 2：bind 失败（端口被占用）

```cpp
// ❌ 端口被占用
bind(server_fd, ...);  // 失败

// ✅ 设置 SO_REUSEADDR
int opt = 1;
setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
bind(server_fd, ...);  // 成功
```

### 陷阱 3：没有检查返回值

```cpp
// ❌ 不检查错误
int sock = socket(AF_INET, SOCK_STREAM, 0);
bind(sock, ...);  // 可能失败
listen(sock, 3);  // 可能失败

// ✅ 检查错误
int sock = socket(AF_INET, SOCK_STREAM, 0);
if (sock < 0) {
    perror("socket");
    return -1;
}

if (bind(sock, ...) < 0) {
    perror("bind");
    close(sock);
    return -1;
}
```

### 陷阱 4：缓冲区溢出

```cpp
// ❌ 缓冲区溢出
char buffer[1024];
read(client_fd, buffer, 2048);  // 危险！

// ✅ 限制读取大小
char buffer[1024];
read(client_fd, buffer, sizeof(buffer));
```

## 核心要点

1. **socket 编程流程**：
   - 服务器：socket → bind → listen → accept → read/write → close
   - 客户端：socket → connect → read/write → close

2. **TCP vs UDP**：
   - TCP：可靠、面向连接、慢
   - UDP：快速、无连接、不可靠

3. **非阻塞 I/O**：
   - 用 `fcntl` 设置 `O_NONBLOCK`

4. **I/O 多路复用**：
   - select：简单，但性能差
   - epoll（Linux）：高效，推荐

5. **RAII 管理 socket**：
   - 自动 close，避免泄漏

6. **错误处理**：
   - 检查所有返回值
   - 用 `perror` 打印错误

7. **实际应用**：
   - HTTP 服务器
   - 聊天服务器
   - 游戏服务器
