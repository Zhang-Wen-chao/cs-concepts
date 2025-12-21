// bRPC Echo 客户端：调用 Protobuf Stub

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/logging.h>
#include <gflags/gflags.h>

#include "echo.pb.h"

DEFINE_string(server, "127.0.0.1:8800", "Server address, e.g. ip:port");
DEFINE_string(message, "Hello bRPC", "Message to echo");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    brpc::Channel channel;
    brpc::ChannelOptions options;
    if (channel.Init(FLAGS_server.c_str(), &options) != 0) {
        LOG(ERROR) << "初始化 Channel 失败";
        return -1;
    }

    example::EchoService_Stub stub(&channel);
    example::EchoRequest request;
    request.set_message(FLAGS_message);

    example::EchoResponse response;
    brpc::Controller cntl;

    stub.Echo(&cntl, &request, &response, nullptr);

    if (cntl.Failed()) {
        LOG(ERROR) << "RPC 失败: " << cntl.ErrorText();
        return -1;
    }

    LOG(INFO) << "服务器响应: " << response.message();
    return 0;
}
