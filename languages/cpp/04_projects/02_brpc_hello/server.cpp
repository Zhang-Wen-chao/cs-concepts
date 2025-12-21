// bRPC Echo 服务：使用 Protobuf 定义的 EchoService

#include <brpc/server.h>
#include <butil/logging.h>
#include <gflags/gflags.h>

#include "echo.pb.h"

DEFINE_int32(port, 8800, "TCP port to listen on");
DEFINE_int32(idle_timeout_s, -1,
             "Close connections if there is no read/write for so many seconds");

class EchoServiceImpl : public example::EchoService {
public:
    void Echo(google::protobuf::RpcController* cntl_base,
              const example::EchoRequest* request,
              example::EchoResponse* response,
              google::protobuf::Closure* done) override {
        brpc::ClosureGuard done_guard(done);

        response->set_message(request->message());

        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        LOG(INFO) << "Echo message: \"" << request->message()
                  << "\" from " << cntl->remote_side();
    }
};

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    brpc::Server server;
    EchoServiceImpl echo_service;

    if (server.AddService(&echo_service,
                          brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "注册 EchoService 失败";
        return -1;
    }

    brpc::ServerOptions options;
    options.idle_timeout_sec = FLAGS_idle_timeout_s;

    if (server.Start(FLAGS_port, &options) != 0) {
        LOG(ERROR) << "端口 " << FLAGS_port << " 启动失败，请尝试 --port=xxxx";
        return -1;
    }

    LOG(INFO) << "Echo server is running on port " << FLAGS_port;
    LOG(INFO) << "Open http://localhost:" << FLAGS_port
              << " to inspect bRPC builtin dashboard";

    server.RunUntilAskedToQuit();
    return 0;
}
