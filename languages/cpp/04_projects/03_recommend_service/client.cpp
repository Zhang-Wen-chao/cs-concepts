// 简单的 bRPC 推荐服务客户端

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/logging.h>
#include <gflags/gflags.h>

#include "recommend.pb.h"

DEFINE_string(server, "127.0.0.1:9000", "Recommend server address");
DEFINE_string(user_id, "alice", "User id in request");
DEFINE_string(items, "itemA:0.2,itemB:0.7", "Comma separated item_id:context_weight pairs");
DEFINE_int32(top_k, 2, "Number of items to request");

namespace {

struct ParsedItem {
    std::string item_id;
    double weight = 0.0;
};

std::vector<ParsedItem> ParseItems(const std::string& spec) {
    std::vector<ParsedItem> items;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        auto pos = token.find(':');
        ParsedItem item;
        if (pos == std::string::npos) {
            item.item_id = token;
            item.weight = 0.0;
        } else {
            item.item_id = token.substr(0, pos);
            item.weight = std::stod(token.substr(pos + 1));
        }
        items.push_back(item);
    }
    return items;
}

}  // namespace

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    brpc::Channel channel;
    brpc::ChannelOptions options;
    if (channel.Init(FLAGS_server.c_str(), &options) != 0) {
        LOG(ERROR) << "初始化 Channel 失败";
        return -1;
    }

    recommend::RecommendService_Stub stub(&channel);
    recommend::RecommendRequest request;
    request.set_user_id(FLAGS_user_id);
    request.set_top_k(FLAGS_top_k);

    auto parsed_items = ParseItems(FLAGS_items);
    if (parsed_items.empty()) {
        LOG(ERROR) << "items 解析为空，检查 --items 参数";
        return -1;
    }
    for (const auto& item : parsed_items) {
        auto* candidate = request.add_candidates();
        candidate->set_item_id(item.item_id);
        candidate->set_context_weight(item.weight);
    }

    recommend::RecommendResponse response;
    brpc::Controller cntl;
    stub.Recommend(&cntl, &request, &response, nullptr);

    if (cntl.Failed()) {
        LOG(ERROR) << "RPC 失败: " << cntl.ErrorText();
        return -1;
    }

    std::cout << "Top-" << response.items_size() << " recommendations for user "
              << FLAGS_user_id << ":\n";
    for (const auto& rec : response.items()) {
        std::cout << "  - " << rec.item_id() << " | score=" << rec.score()
                  << " | reason=" << rec.reason() << "\n";
    }
    return 0;
}
