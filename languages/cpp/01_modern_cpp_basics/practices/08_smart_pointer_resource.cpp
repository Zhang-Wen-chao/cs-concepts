/**
 * 实践项目 2：用智能指针管理资源的小程序
 *
 * 编译：g++ -std=c++17 08_smart_pointer_resource.cpp -o smart_pointer_resource
 * 运行：./smart_pointer_resource
 */

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Resource {
public:
    explicit Resource(std::string name)
        : name_(std::move(name)), payload_(std::make_unique<std::vector<int>>(1024, 1)) {
        std::cout << "[Resource] acquired " << name_ << std::endl;
    }

    ~Resource() {
        std::cout << "[Resource] released " << name_ << std::endl;
    }

    void Use() const {
        const auto checksum = payload_->size();
        std::cout << "    using resource " << name_ << ", checksum " << checksum << std::endl;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::unique_ptr<std::vector<int>> payload_;
};

class ResourceManager {
public:
    std::shared_ptr<Resource> Acquire(const std::string& name) {
        CollectGarbage();
        if (auto existing = GetAlive(name)) {
            std::cout << "[Manager] reuse cached resource: " << name << std::endl;
            return existing;
        }
        auto deleter = [name](Resource* resource) {
            std::cout << "[Manager] custom delete for " << name << std::endl;
            delete resource;
        };
        auto resource = std::shared_ptr<Resource>(new Resource(name), std::move(deleter));
        cache_[name] = resource;
        return resource;
    }

    void CollectGarbage() {
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (it->second.expired()) {
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void PrintStats() const {
        size_t alive = 0;
        for (const auto& [_, weak_resource] : cache_) {
            if (!weak_resource.expired()) {
                ++alive;
            }
        }
        std::cout << "[Manager] cache slots: " << cache_.size()
                  << ", alive resources: " << alive << std::endl;
    }

private:
    std::shared_ptr<Resource> GetAlive(const std::string& name) {
        auto it = cache_.find(name);
        if (it == cache_.end()) {
            return nullptr;
        }
        return it->second.lock();
    }

    std::unordered_map<std::string, std::weak_ptr<Resource>> cache_;
};

class Worker {
public:
    Worker(std::string name, std::shared_ptr<Resource> resource)
        : name_(std::move(name)), resource_(std::move(resource)) {}

    void Process() const {
        if (auto res = resource_) {
            std::cout << "[" << name_ << "] start using " << res->name() << std::endl;
            res->Use();
            std::cout << "[" << name_ << "] done" << std::endl;
        }
    }

private:
    std::string name_;
    std::shared_ptr<Resource> resource_;
};

void RunSmartPointerDemo() {
    std::cout << "\n=== 智能指针资源管理示例 ===" << std::endl;
    auto manager = std::make_unique<ResourceManager>();

    auto dataset = manager->Acquire("dataset.bin");
    {
        Worker loader("Loader", dataset);
        loader.Process();
    }

    {
        auto shared_config = manager->Acquire("config.json");
        Worker validator("Validator", shared_config);
        Worker executor("Executor", shared_config);
        validator.Process();
        executor.Process();
        std::cout << "config.json use_count = " << shared_config.use_count() << std::endl;
    }

    manager->PrintStats();

    std::weak_ptr<Resource> weak_link;
    {
        auto temp = manager->Acquire("temporary.cache");
        weak_link = temp;
        Worker reporter("Reporter", temp);
        reporter.Process();
        std::cout << "temporary.cache use_count (inside scope) = " << temp.use_count() << std::endl;
    }

    if (weak_link.expired()) {
        std::cout << "temporary.cache 已经释放" << std::endl;
    } else {
        std::cout << "temporary.cache 仍在内存，自动清理后释放" << std::endl;
    }

    manager->CollectGarbage();
    manager->PrintStats();
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "现代 C++ 实践 2：智能指针资源管理" << std::endl;
    std::cout << "============================================" << std::endl;

    RunSmartPointerDemo();

    std::cout << "\n示例执行完毕！" << std::endl;
    return 0;
}
