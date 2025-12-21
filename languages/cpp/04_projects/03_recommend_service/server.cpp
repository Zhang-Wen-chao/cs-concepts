// bRPC 推荐服务：将简单打分模型暴露为 Recommend RPC

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <brpc/server.h>
#include <butil/logging.h>
#include <gflags/gflags.h>

#include "recommend.pb.h"

DEFINE_int32(port, 9000, "Server port");
DEFINE_string(model_path, "data/model_config.csv", "Model config CSV (type,key,value)");
DEFINE_int32(default_top_k, 3, "Fallback top_k when request.top_k <= 0");

namespace {

class PreferenceModel {
public:
    PreferenceModel() = default;

    bool Load(const std::string& path) {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            LOG(WARNING) << "无法打开模型文件 " << path << ", 使用内置默认权重";
            LoadDefaults();
            return false;
        }

        std::string header;
        std::getline(ifs, header);  // 跳过标题
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) {
                continue;
            }
            std::istringstream iss(line);
            std::string type, key, value_str;
            if (!std::getline(iss, type, ',')) {
                continue;
            }
            if (!std::getline(iss, key, ',')) {
                continue;
            }
            if (!std::getline(iss, value_str, ',')) {
                continue;
            }
            double value = std::stod(value_str);
            if (type == "user") {
                user_bias_[key] = value;
            } else if (type == "item") {
                item_bias_[key] = value;
            }
        }

        if (user_bias_.empty() && item_bias_.empty()) {
            LOG(WARNING) << "模型文件为空，使用默认参数";
            LoadDefaults();
        }
        return true;
    }

    double UserBias(const std::string& user) const {
        auto it = user_bias_.find(user);
        return it == user_bias_.end() ? 0.0 : it->second;
    }

    double ItemBias(const std::string& item) const {
        auto it = item_bias_.find(item);
        return it == item_bias_.end() ? 0.0 : it->second;
    }

private:
    void LoadDefaults() {
        user_bias_ = { {"alice", 0.8}, {"bob", -0.3}, {"charlie", 0.1} };
        item_bias_ = { {"itemA", 1.0}, {"itemB", 0.5}, {"itemC", -0.2}, {"itemD", 0.7} };
    }

    std::unordered_map<std::string, double> user_bias_;
    std::unordered_map<std::string, double> item_bias_;
};

struct ScoredItem {
    std::string item_id;
    double score = 0.0;
    std::string reason;
};

class RecommendServiceImpl : public recommend::RecommendService {
public:
    explicit RecommendServiceImpl(const PreferenceModel* model) : model_(model) {}

    void Recommend(google::protobuf::RpcController* cntl_base,
                   const recommend::RecommendRequest* request,
                   recommend::RecommendResponse* response,
                   google::protobuf::Closure* done) override {
        brpc::ClosureGuard done_guard(done);

        if (request->candidates().empty()) {
            LOG(WARNING) << "收到空候选的请求 user=" << request->user_id();
            return;
        }

        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        LOG(INFO) << "Recommend 请求 user=" << request->user_id()
                  << " candidates=" << request->candidates_size()
                  << " from " << cntl->remote_side();

        std::vector<ScoredItem> scored;
        scored.reserve(request->candidates_size());
        double user_bias = model_->UserBias(request->user_id());

        for (const auto& cand : request->candidates()) {
            double item_bias = model_->ItemBias(cand.item_id());
            double score = user_bias + item_bias + cand.context_weight();
            std::ostringstream reason;
            reason << "user_bias=" << user_bias
                   << ",item_bias=" << item_bias
                   << ",context=" << cand.context_weight();
            scored.push_back({cand.item_id(), score, reason.str()});
        }

        int requested_k = request->top_k();
        if (requested_k <= 0) {
            requested_k = FLAGS_default_top_k;
        }
        requested_k = std::min(requested_k, static_cast<int>(scored.size()));

        std::partial_sort(scored.begin(), scored.begin() + requested_k, scored.end(),
                          [](const ScoredItem& lhs, const ScoredItem& rhs) {
                              return lhs.score > rhs.score;
                          });

        for (int i = 0; i < requested_k; ++i) {
            auto* rec = response->add_items();
            rec->set_item_id(scored[i].item_id);
            rec->set_score(scored[i].score);
            rec->set_reason(scored[i].reason);
        }
    }

private:
    const PreferenceModel* model_;
};

}  // namespace

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    PreferenceModel model;
    model.Load(FLAGS_model_path);

    brpc::Server server;
    RecommendServiceImpl service(&model);
    if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        LOG(ERROR) << "注册 RecommendService 失败";
        return -1;
    }

    brpc::ServerOptions options;
    if (server.Start(FLAGS_port, &options) != 0) {
        LOG(ERROR) << "启动失败，端口: " << FLAGS_port;
        return -1;
    }

    LOG(INFO) << "Recommend server is running on port " << FLAGS_port;
    LOG(INFO) << "Visit http://localhost:" << FLAGS_port << " for metrics";

    server.RunUntilAskedToQuit();
    return 0;
}
