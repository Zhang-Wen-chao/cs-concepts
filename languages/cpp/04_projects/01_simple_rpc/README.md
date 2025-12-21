# 简化版 RPC 框架

> 从零实现一个简单的 RPC（Remote Procedure Call）框架

## 什么是 RPC？

**RPC = Remote Procedure Call（远程过程调用）**

**核心思想**：像调用本地函数一样调用远程服务

```cpp
// 客户端代码（看起来像本地调用）
int result = calculator.add(3, 5);  // 实际在另一台机器上执行
```

**背后发生的事**：
```
客户端                      网络                      服务器
  |                          |                          |
  | 1. 调用 add(3, 5)        |                          |
  | 2. 序列化参数            |                          |
  |------------------------->|                          |
  |    发送请求              |                          |
  |                          |------------------------->|
  |                          |    3. 接收请求           |
  |                          |    4. 反序列化参数       |
  |                          |    5. 执行 add(3, 5)     |
  |                          |<-------------------------|
  |<-------------------------|    6. 返回结果           |
  | 7. 反序列化结果          |                          |
  | 8. 返回 8                |                          |
```

## 项目目标

实现一个简化版 RPC 框架，包括：
1. **网络层**：TCP socket 通信
2. **序列化层**：简单的二进制序列化
3. **服务注册**：注册可调用的函数
4. **客户端**：远程调用接口

## 项目结构

```
01_simple_rpc/
├── README.md           # 项目说明
├── rpc_server.h        # 服务器接口
├── rpc_server.cpp      # 服务器实现
├── rpc_client.h        # 客户端接口
├── rpc_client.cpp      # 客户端实现
├── rpc_protocol.h      # 协议定义
├── example_server.cpp  # 示例服务器
└── example_client.cpp  # 示例客户端
```

## 协议设计

**请求格式**：
```
[4字节：消息长度][4字节：函数ID][N字节：参数数据]
```

**响应格式**：
```
[4字节：消息长度][4字节：状态码][N字节：返回值数据]
```

## 核心功能

### 1. 服务器

```cpp
RpcServer server(8080);

// 注册函数
server.register_function("add", [](int a, int b) {
    return a + b;
});

// 启动服务
server.start();
```

### 2. 客户端

```cpp
RpcClient client("127.0.0.1", 8080);

// 远程调用
int result = client.call<int>("add", 3, 5);
std::cout << "结果: " << result << "\n";  // 8
```

## 实现步骤

### 阶段 1：基础协议
- [x] 定义消息格式
- [x] 实现序列化/反序列化

### 阶段 2：服务器
- [x] 创建 TCP 服务器
- [x] 注册函数
- [x] 处理请求

### 阶段 3：客户端
- [x] 创建 TCP 客户端
- [x] 发送请求
- [x] 接收响应

### 阶段 4：测试
- [x] 实现示例服务
- [x] 测试远程调用

## 技术要点

**1. 网络层**：
- TCP socket（可靠传输）
- 先发长度，再发数据

**2. 序列化**：
- 简单二进制格式
- 支持 int、string

**3. 函数注册**：
- 用 `std::function` 存储函数
- 用 `unordered_map` 映射函数名

**4. 线程安全**：
- 多个客户端同时连接
- 用线程池处理请求

## 使用示例

### 服务器

```cpp
#include "rpc_server.h"

int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

int main() {
    RpcServer server(8080);

    // 注册服务
    server.register_function("add", add);
    server.register_function("multiply", multiply);

    std::cout << "RPC 服务器启动在 8080\n";
    server.start();  // 阻塞

    return 0;
}
```

### 客户端

```cpp
#include "rpc_client.h"

int main() {
    RpcClient client("127.0.0.1", 8080);

    // 远程调用
    int sum = client.call<int>("add", 10, 20);
    std::cout << "10 + 20 = " << sum << "\n";

    int product = client.call<int>("multiply", 5, 6);
    std::cout << "5 * 6 = " << product << "\n";

    return 0;
}
```

### 运行

```bash
# 编译
g++ -std=c++17 rpc_server.cpp example_server.cpp -o server
g++ -std=c++17 rpc_client.cpp example_client.cpp -o client

# 运行服务器
./server

# 运行客户端（另一个终端）
./client
```

## 输出示例

**服务器**：
```
RPC 服务器启动在 8080
等待连接...
客户端连接: 127.0.0.1
收到请求: add(10, 20)
返回结果: 30
收到请求: multiply(5, 6)
返回结果: 30
```

**客户端**：
```
连接到服务器 127.0.0.1:8080
10 + 20 = 30
5 * 6 = 30
```

## 扩展方向

**1. 支持更多类型**：
- double、string、vector
- 自定义结构体

**2. 异步调用**：
- 非阻塞 I/O
- Future/Promise

**3. 超时机制**：
- 请求超时
- 重试

**4. 负载均衡**：
- 多服务器
- 轮询/随机

**5. 服务发现**：
- 注册中心
- 动态发现服务

## 与工业界 RPC 对比

| 特性 | 简化版 | gRPC | bRPC |
|------|--------|------|------|
| 序列化 | 二进制 | Protobuf | Protobuf |
| 协议 | TCP | HTTP/2 | 多种 |
| 性能 | 中 | 高 | 极高 |
| 功能 | 基础 | 完整 | 完整 |

## 核心收获

1. **理解 RPC 原理**：
   - 网络通信
   - 序列化/反序列化
   - 函数调用抽象

2. **综合运用技术**：
   - Socket 编程
   - 多线程
   - 模板编程

3. **为学习 bRPC 打基础**：
   - 理解 RPC 框架设计
   - 知道工业级框架的优势
