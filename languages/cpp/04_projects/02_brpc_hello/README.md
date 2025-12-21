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

### macOS（推荐：使用 Homebrew）

```bash
# 直接安装 bRPC 及其依赖（protobuf 29 + Abseil）
brew install brpc protobuf@29 abseil
```

### 验证安装

```bash
# 检查头文件/库文件（brpc）
ls /opt/homebrew/include/brpc
ls /opt/homebrew/lib/libbrpc.*

# 检查 protobuf/absl 是否准备好
ls /opt/homebrew/opt/protobuf@29/include/google/protobuf
ls /opt/homebrew/opt/abseil/lib
```

## 项目结构

```
02_brpc_hello/
├── README.md                  # 项目说明
├── Makefile                   # 编译脚本
├── server.cpp                 # 可运行的 bRPC Echo 服务器
├── client.cpp                 # 对应的 Echo 客户端
├── echo.proto                 # Protobuf 服务定义
├── echo_server_example.cpp    # 服务器示例代码（学习用）
└── echo_client_example.cpp    # 客户端示例代码（学习用）
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
#include <butil/logging.h>
#include <gflags/gflags.h>
#include "echo.pb.h"

DEFINE_int32(port, 8800, "TCP Port of this server");

class EchoServiceImpl : public example::EchoService {
public:

    void Echo(google::protobuf::RpcController* cntl_base,
              const example::EchoRequest* request,
              example::EchoResponse* response,
              google::protobuf::Closure* done) override {

        brpc::ClosureGuard done_guard(done);  // RAII 自动调用 done

        // 业务逻辑：返回相同消息
        response->set_message(request->message());

        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        LOG(INFO) << "收到: " << request->message()
                  << " from " << cntl->remote_side();
    }
};

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    brpc::Server server;

    // 注册服务
    EchoServiceImpl echo_service;
    if (server.AddService(&echo_service,
                          brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "注册服务失败";
        return -1;
    }

    // 启动服务器
    brpc::ServerOptions options;
    if (server.Start(FLAGS_port, &options) != 0) {
        LOG(ERROR) << "端口 " << FLAGS_port << " 启动失败";
        return -1;
    }

    LOG(INFO) << "Echo server running at http://localhost:" << FLAGS_port;
    server.RunUntilAskedToQuit();
    return 0;
}
```

## 客户端实现

```cpp
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/logging.h>
#include <gflags/gflags.h>
#include "echo.pb.h"

DEFINE_string(server, "127.0.0.1:8800", "Server address, e.g. ip:port");
DEFINE_string(message, "Hello bRPC", "Message to echo");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    brpc::Channel channel;

    // 初始化 channel
    brpc::ChannelOptions options;
    if (channel.Init(FLAGS_server.c_str(), &options) != 0) {
        LOG(ERROR) << "初始化 channel 失败";
        return -1;
    }

    // 创建 stub
    example::EchoService_Stub stub(&channel);

    // 发送请求
    example::EchoRequest request;
    request.set_message(FLAGS_message);

    example::EchoResponse response;
    brpc::Controller cntl;

    stub.Echo(&cntl, &request, &response, nullptr);

    if (cntl.Failed()) {
        LOG(ERROR) << "RPC 失败: " << cntl.ErrorText();
        return 1;
    }

    LOG(INFO) << "收到响应: " << response.message();
    return 0;
}
```

## 编译和运行

```bash
# 生成 Protobuf + 编译 server/client
make

# 启动服务器（默认 8800 端口，可通过 --port 调整）
./server --port=8800

# 新开一个终端，调用客户端
./client --server=127.0.0.1:8800 --message="Hello bRPC"

# 在浏览器中访问
# http://localhost:8800 - 查看 bRPC 内置监控页面
# http://localhost:8800/status - 查看服务状态
```

**成功标志**：
- ✅ bRPC 成功编译
- ✅ 服务器成功启动 + 客户端成功收到响应
- ✅ 可以访问内置监控页面 (`/`, `/vars`, `/status`)

**说明**：
- `server.cpp` / `client.cpp` 是主线可运行代码
- `echo_server_example.cpp` 和 `echo_client_example.cpp` 是扩展示例，展示更多注释和额外特性

**学习建议**：
1. 先运行 `server`/`client` 体验基础流程
2. 再阅读示例代码理解更多 bRPC 细节
3. 参考 [bRPC 官方示例](https://github.com/apache/brpc/tree/master/example)
4. 实际项目中使用 bRPC 时，确保 Protobuf 版本匹配

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
// 访问 http://localhost:<port>
// 自动提供服务信息、统计数据
```

### 2. 性能监控

```cpp
// 访问 http://localhost:<port>/vars
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
