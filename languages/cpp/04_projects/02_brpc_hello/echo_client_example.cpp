// bRPC Echo 客户端示例
// 注意：需要先安装 bRPC 才能编译运行
//
// 编译：见 INSTALL.md
// 运行：./echo_client

#include <brpc/channel.h>
#include <gflags/gflags.h>
#include "echo.pb.h"

DEFINE_string(server, "127.0.0.1:8080", "IP Address of server");
DEFINE_int32(timeout_ms, 1000, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries (not including the first RPC)");

int main(int argc, char* argv[]) {
    // 解析命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // 创建 Channel（连接）
    brpc::Channel channel;

    // 配置 Channel 选项
    brpc::ChannelOptions options;
    options.timeout_ms = FLAGS_timeout_ms;
    options.max_retry = FLAGS_max_retry;

    // 初始化 Channel
    if (channel.Init(FLAGS_server.c_str(), &options) != 0) {
        LOG(ERROR) << "初始化 Channel 失败";
        return -1;
    }

    LOG(INFO) << "连接到服务器: " << FLAGS_server;

    // 创建 Stub（桩）
    example::EchoService_Stub stub(&channel);

    // 准备请求
    example::EchoRequest request;
    request.set_message("Hello bRPC");

    // 准备响应
    example::EchoResponse response;

    // 创建 Controller
    brpc::Controller cntl;

    // 发起 RPC 调用（同步）
    LOG(INFO) << "发送请求: " << request.message();
    stub.Echo(&cntl, &request, &response, nullptr);

    // 检查调用是否成功
    if (cntl.Failed()) {
        LOG(ERROR) << "RPC 失败: " << cntl.ErrorText();
        return -1;
    }

    // 打印响应
    LOG(INFO) << "收到响应: " << response.message();
    LOG(INFO) << "延迟: " << cntl.latency_us() << " μs";

    // 测试多次调用
    LOG(INFO) << "\n测试多次调用:";
    for (int i = 1; i <= 5; ++i) {
        example::EchoRequest req;
        req.set_message("消息 " + std::to_string(i));

        example::EchoResponse resp;
        brpc::Controller ctrl;

        stub.Echo(&ctrl, &req, &resp, nullptr);

        if (!ctrl.Failed()) {
            LOG(INFO) << "  [" << i << "] "
                      << req.message() << " -> "
                      << resp.message()
                      << " (延迟: " << ctrl.latency_us() << " μs)";
        }
    }

    LOG(INFO) << "测试完成";

    return 0;
}

/*
 * bRPC 客户端关键点：
 *
 * 1. Channel（连接）：
 *    - 初始化：Channel.Init(server_address, options)
 *    - 配置：超时、重试、负载均衡等
 *    - 复用：一个 Channel 可发起多次 RPC
 *
 * 2. Stub（桩）：
 *    - 由 Protobuf 自动生成
 *    - 提供类型安全的 RPC 调用接口
 *
 * 3. Controller：
 *    - 设置请求参数（超时、附件等）
 *    - 获取响应信息（延迟、错误等）
 *
 * 4. 同步 vs 异步：
 *    - 同步：传 nullptr 给 done 参数，阻塞等待
 *    - 异步：传 Closure 给 done，立即返回
 *
 * 5. 错误处理：
 *    - cntl.Failed() 检查是否失败
 *    - cntl.ErrorText() 获取错误信息
 *
 * 6. 性能监控：
 *    - cntl.latency_us() 获取延迟
 *    - bRPC 自动收集统计数据
 *
 * 7. 高级特性：
 *    - 连接池（自动管理）
 *    - 超时重试
 *    - 负载均衡
 *    - 熔断降级
 */
