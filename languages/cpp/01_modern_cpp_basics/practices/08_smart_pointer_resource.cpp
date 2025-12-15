// 智能指针资源管理（最小示例）
// 编译：g++ -std=c++17 08_smart_pointer_resource.cpp -o resource

#include <iostream>
#include <memory>
#include <vector>

class Resource {
    int id_;
public:
    Resource(int id) : id_(id) { std::cout << "Resource(" << id_ << ")\n"; }
    ~Resource() { std::cout << "~Resource(" << id_ << ")\n"; }
    void use() { std::cout << "Using resource " << id_ << "\n"; }
};

// 工厂函数返回 unique_ptr
std::unique_ptr<Resource> create_resource(int id) {
    return std::make_unique<Resource>(id);
}

int main() {
    // 1. unique_ptr（独占）
    {
        auto r1 = create_resource(1);
        r1->use();
    }  // 自动释放

    // 2. shared_ptr（共享）
    {
        auto r2 = std::make_shared<Resource>(2);
        {
            auto r3 = r2;  // 共享所有权
            std::cout << "ref count: " << r2.use_count() << "\n";  // 2
        }  // r3 销毁，引用计数-1
        std::cout << "ref count: " << r2.use_count() << "\n";  // 1
    }  // r2 销毁，资源释放

    // 3. 容器中管理资源
    {
        std::vector<std::unique_ptr<Resource>> resources;
        resources.push_back(std::make_unique<Resource>(3));
        resources.push_back(std::make_unique<Resource>(4));

        for (auto& r : resources) {
            r->use();
        }
    }  // 自动释放所有资源

    std::cout << "All resources released\n";
}
