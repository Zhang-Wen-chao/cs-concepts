// bRPC Echo 服务器示例
// 注意：需要先安装 bRPC 才能编译运行
//
// 编译：见 INSTALL.md
// 运行：./echo_server

#include <brpc/server.h>
#include <gflags/gflags.h>
#include "echo.pb.h"

DEFINE_int32(port, 8080, "TCP Port of this server");
DEFINE_int32(idle_timeout_s, -1, "Connection will be closed if there is no read/write operations during the last `idle_timeout_s`");

// 实现 Echo 服务
class EchoServiceImpl : public example::EchoService {
public:
    EchoServiceImpl() = default;
    ~EchoServiceImpl() = default;

    // 实现 Echo RPC 方法
    void Echo(google::protobuf::RpcController* cntl_base,
              const example::EchoRequest* request,
              example::EchoResponse* response,
              google::protobuf::Closure* done) override {

        // RAII：确保调用 done->Run()
        brpc::ClosureGuard done_guard(done);

        // 获取 bRPC 的 Controller（扩展功能）
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);

        // 打印请求信息
        LOG(INFO) << "收到请求: " << request->message()
                  << " 来自 " << cntl->remote_side();

        // 业务逻辑：简单地返回相同消息
        response->set_message(request->message());
    }
};

int main(int argc, char* argv[]) {
    // 解析命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 创建服务器
    brpc::Server server;

    // 创建并注册 Echo 服务
    EchoServiceImpl echo_service_impl;
    if (server.AddService(&echo_service_impl,
                          brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "注册服务失败";
        return -1;
    }

    // 配置服务器选项
    brpc::ServerOptions options;
    options.idle_timeout_sec = FLAGS_idle_timeout_s;

    // 启动服务器
    if (server.Start(FLAGS_port, &options) != 0) {
        LOG(ERROR) << "启动服务器失败";
        return -1;
    }

    LOG(INFO) << "Echo 服务器启动在端口 " << FLAGS_port;
    LOG(INFO) << "访问 http://localhost:" << FLAGS_port << " 查看服务信息";

    // 等待退出信号（Ctrl+C）
    server.RunUntilAskedToQuit();

    return 0;
}

/*
 * bRPC 服务器关键点：
 *
 * 1. Service 实现：
 *    - 继承 Protobuf 生成的服务基类
 *    - 实现 RPC 方法
 *    - 使用 ClosureGuard 自动调用 done
 *
 * 2. Controller：
 *    - 获取请求信息（客户端地址、超时等）
 *    - 设置响应信息（错误码、附件等）
 *
 * 3. Server：
 *    - AddService() 注册服务
 *    - Start() 启动服务器
 *    - RunUntilAskedToQuit() 阻塞等待
 *
 * 4. 内置监控：
 *    - http://localhost:8080 - 服务信息
 *    - http://localhost:8080/vars - 性能统计
 *    - http://localhost:8080/status - 服务状态
 *
 * 5. 高级特性：
 *    - 多线程处理（自动）
 *    - 连接复用
 *    - 负载均衡
 *    - 服务发现
 */
