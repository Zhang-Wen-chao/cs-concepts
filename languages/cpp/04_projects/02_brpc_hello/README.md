# bRPC Hello World

> 学习使用百度开源的工业级 RPC 框架 bRPC

## 什么是 bRPC？

**bRPC = Baidu RPC**

百度开源的高性能 RPC 框架，在百度内部广泛使用。

**特点**：
- 🚀 **极高性能**：单机百万 QPS
- 🔧 **功能丰富**：支持多种协议（HTTP、Redis、Thrift 等）
- 📊 **内置监控**：自带性能分析工具
- 🎯 **生产级别**：百度内部久经考验

**对比我们的简化版 RPC**：

| 特性 | 简化版 RPC | bRPC |
|------|-----------|------|
| 性能 | 中 | 极高（百万 QPS）|
| 协议 | 自定义 | Protobuf + 多种 |
| 功能 | 基础 | 完整（负载均衡、服务发现）|
| 监控 | 无 | 内置 |
| 线程模型 | 简单 | 复杂优化 |

## 项目目标

使用 bRPC 实现一个简单的 Echo 服务：
1. 客户端发送消息
2. 服务器返回相同消息

## 安装 bRPC

### macOS

```bash
# 安装依赖
brew install protobuf leveldb gflags openssl

# 克隆 bRPC
git clone https://github.com/apache/brpc.git
cd brpc

# 编译
mkdir build && cd build
cmake ..
make -j8

# 安装
sudo make install
```

### 验证安装

```bash
# 检查头文件
ls /usr/local/include/brpc

# 检查库文件
ls /usr/local/lib/libbrpc.*
```

## 项目结构

```
02_brpc_hello/
├── README.md           # 项目说明
├── echo.proto          # Protobuf 服务定义
├── echo_server.cpp     # 服务器
├── echo_client.cpp     # 客户端
└── Makefile            # 编译脚本
```

## Protobuf 定义

```protobuf
// echo.proto
syntax = "proto3";

package example;

// 请求消息
message EchoRequest {
    string message = 1;
}

// 响应消息
message EchoResponse {
    string message = 1;
}

// Echo 服务
service EchoService {
    rpc Echo(EchoRequest) returns (EchoResponse);
}
```

## 服务器实现

```cpp
#include <brpc/server.h>
#include "echo.pb.h"

// 实现 Echo 服务
class EchoServiceImpl : public example::EchoService {
public:
    void Echo(google::protobuf::RpcController* cntl_base,
              const example::EchoRequest* request,
              example::EchoResponse* response,
              google::protobuf::Closure* done) override {

        brpc::ClosureGuard done_guard(done);  // RAII 自动调用 done

        // 业务逻辑：返回相同消息
        response->set_message(request->message());

        std::cout << "收到: " << request->message() << "\n";
    }
};

int main() {
    brpc::Server server;

    // 注册服务
    EchoServiceImpl echo_service;
    server.AddService(&echo_service, brpc::SERVER_DOESNT_OWN_SERVICE);

    // 启动服务器
    brpc::ServerOptions options;
    server.Start(8080, &options);

    server.RunUntilAskedToQuit();
    return 0;
}
```

## 客户端实现

```cpp
#include <brpc/channel.h>
#include "echo.pb.h"

int main() {
    brpc::Channel channel;

    // 初始化 channel
    brpc::ChannelOptions options;
    channel.Init("127.0.0.1:8080", &options);

    // 创建 stub
    example::EchoService_Stub stub(&channel);

    // 发送请求
    example::EchoRequest request;
    request.set_message("Hello bRPC");

    example::EchoResponse response;
    brpc::Controller cntl;

    stub.Echo(&cntl, &request, &response, nullptr);

    if (cntl.Failed()) {
        std::cerr << "RPC 失败: " << cntl.ErrorText() << "\n";
        return 1;
    }

    std::cout << "收到响应: " << response.message() << "\n";
    return 0;
}
```

## 编译和运行

```bash
# 生成 Protobuf 代码
protoc --cpp_out=. echo.proto

# 编译
make

# 运行服务器（终端1）
./echo_server

# 运行客户端（终端2）
./echo_client
```

## bRPC 核心概念

### 1. Service（服务）

```cpp
// 继承 Protobuf 生成的服务基类
class MyService : public example::MyService {
    // 实现 RPC 方法
};
```

### 2. Server（服务器）

```cpp
brpc::Server server;
server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE);
server.Start(port, &options);
```

### 3. Channel（客户端连接）

```cpp
brpc::Channel channel;
channel.Init("host:port", &options);
```

### 4. Controller（控制器）

```cpp
brpc::Controller cntl;
cntl.set_timeout_ms(100);  // 设置超时
// 调用后检查
if (cntl.Failed()) {
    // 处理错误
}
```

### 5. Closure（回调）

```cpp
// 同步调用：传 nullptr
stub.Echo(&cntl, &request, &response, nullptr);

// 异步调用：传回调
google::protobuf::Closure* done = ...;
stub.Echo(&cntl, &request, &response, done);
```

## bRPC 高级特性

### 1. 内置 HTTP 服务

```cpp
// 访问 http://localhost:8080
// 自动提供服务信息、统计数据
```

### 2. 性能监控

```cpp
// 访问 http://localhost:8080/vars
// 查看 QPS、延迟、错误率等
```

### 3. 负载均衡

```cpp
brpc::ChannelOptions options;
options.load_balancer_name = "random";  // 随机
// 或 "rr"（轮询）、"c_hash"（一致性哈希）
```

### 4. 超时重试

```cpp
brpc::Controller cntl;
cntl.set_timeout_ms(100);     // 超时 100ms
cntl.set_max_retry(3);        // 最多重试 3 次
```

### 5. 异步调用

```cpp
// 异步调用，不阻塞
stub.Echo(&cntl, &request, &response,
    google::protobuf::NewCallback(&OnRpcDone, ...));

// 继续做其他事...
```

## 与简化版 RPC 对比

| 功能 | 简化版 | bRPC |
|------|--------|------|
| 协议定义 | 手写 | Protobuf（自动生成）|
| 序列化 | 手写二进制 | Protobuf（高效）|
| 网络层 | 简单 TCP | 高性能异步 I/O |
| 线程模型 | 单线程 | 多线程 + 协程 |
| 监控 | 无 | 内置完整监控 |
| 性能 | ~1K QPS | ~100万 QPS |
| 代码量 | 多 | 少（框架自动处理）|

## 核心收获

1. **理解工业级 RPC 框架**：
   - 自动代码生成（Protobuf）
   - 高性能网络 I/O
   - 完整的监控和管理

2. **对比学习**：
   - 简化版 RPC：理解原理
   - bRPC：理解工业实践

3. **实际应用**：
   - 微服务架构
   - 分布式系统
   - 高性能服务

## 下一步

- 学习更多 bRPC 特性
- 实现推荐服务（把深度学习模型包装成 RPC 服务）
- 了解微服务架构

## 参考资源

- [bRPC 官方文档](https://brpc.apache.org/)
- [bRPC GitHub](https://github.com/apache/brpc)
- [Protobuf 文档](https://protobuf.dev/)
