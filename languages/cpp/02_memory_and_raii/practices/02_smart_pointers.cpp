// 智能指针核心示例
// 编译：g++ -std=c++17 02_smart_pointers.cpp -o smart_ptr

#include <iostream>
#include <memory>
#include <vector>

struct Node {
    int val;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // 用 weak_ptr 打破循环引用

    Node(int v) : val(v) { std::cout << "Node(" << val << ")\n"; }
    ~Node() { std::cout << "~Node(" << val << ")\n"; }
};

int main() {
    // 1. unique_ptr（独占）
    auto p1 = std::make_unique<int>(42);
    std::cout << "unique_ptr: " << *p1 << "\n";

    // 移动所有权
    auto p2 = std::move(p1);
    std::cout << "p1 is null: " << (p1 == nullptr) << "\n";

    // 2. shared_ptr（共享）
    auto sp1 = std::make_shared<int>(100);
    auto sp2 = sp1;  // 拷贝，引用计数+1
    std::cout << "ref count: " << sp1.use_count() << "\n";  // 2

    // 3. weak_ptr（打破循环引用）
    auto n1 = std::make_shared<Node>(1);
    auto n2 = std::make_shared<Node>(2);

    n1->next = n2;   // 强引用
    n2->prev = n1;   // 弱引用（打破循环）

    // 4. 容器中存储智能指针
    std::vector<std::unique_ptr<int>> vec;
    vec.push_back(std::make_unique<int>(10));
    vec.push_back(std::make_unique<int>(20));

    std::cout << "Container: ";
    for (const auto& p : vec) {
        std::cout << *p << " ";
    }
    std::cout << "\n";

    return 0;
}
