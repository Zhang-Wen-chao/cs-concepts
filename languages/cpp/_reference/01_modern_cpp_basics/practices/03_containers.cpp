// 容器核心示例
// 编译：g++ -std=c++17 03_containers.cpp -o containers

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

int main() {
    // 1. vector（默认选择）
    std::vector<int> v = {1, 2, 3};
    v.push_back(4);
    v.emplace_back(5);
    std::cout << "vector[0]: " << v[0] << "\n";

    // 遍历
    for (const auto& x : v) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // 2. unordered_map（键值查找）
    std::unordered_map<std::string, int> m;
    m["apple"] = 5;
    m["banana"] = 3;

    std::cout << "apple: " << m["apple"] << "\n";

    if (m.count("apple")) {
        std::cout << "Has apple\n";
    }

    // 3. unordered_set（去重）
    std::unordered_set<int> s = {1, 2, 3, 2, 1};  // 自动去重
    s.insert(4);

    std::cout << "set size: " << s.size() << "\n";  // 4

    if (s.count(2)) {
        std::cout << "Has 2\n";
    }

    return 0;
}
