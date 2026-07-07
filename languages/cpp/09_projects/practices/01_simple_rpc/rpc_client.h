#pragma once

#include "rpc_protocol.h"
#include <string>

namespace simple_rpc {

class RpcClient {
    std::string host_;
    int port_;
    int sock_fd_;

public:
    RpcClient(const std::string& host, int port);
    ~RpcClient();

    // 连接服务器
    bool connect();

    // 远程调用（目前只支持 int(int, int) 类型）
    int call(const std::string& function_name, int arg1, int arg2);

private:
    // 发送请求
    bool send_request(uint32_t function_id, int arg1, int arg2);

    // 接收响应
    int receive_response();
};

} // namespace simple_rpc
