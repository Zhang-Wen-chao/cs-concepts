// 模板核心示例
// 编译：g++ -std=c++17 06_templates_basics.cpp -o templates

#include <iostream>
#include <vector>

// 函数模板
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

// 类模板
template<typename T>
class Box {
    T value_;
public:
    Box(T v) : value_(v) {}
    T get() const { return value_; }
    void set(T v) { value_ = v; }
};

// 多参数模板
template<typename T, typename U>
auto add(T a, U b) {
    return a + b;
}

// 通用打印
template<typename Container>
void print_all(const Container& c) {
    for (const auto& item : c) {
        std::cout << item << " ";
    }
    std::cout << "\n";
}

// 变长模板（可变参数）
template<typename... Args>
void print(Args... args) {
    (std::cout << ... << args) << "\n";  // C++17 折叠表达式
}

int main() {
    // 1. 函数模板
    std::cout << "max(3, 5): " << max(3, 5) << "\n";
    std::cout << "max(1.5, 2.5): " << max(1.5, 2.5) << "\n";

    // 2. 类模板
    Box<int> b1(42);
    Box<std::string> b2("hello");
    std::cout << "Box: " << b1.get() << "\n";

    // 3. 多参数
    std::cout << "add(3, 1.5): " << add(3, 1.5) << "\n";

    // 4. 通用算法
    std::vector<int> v = {1, 2, 3, 4, 5};
    print_all(v);

    // 5. 变长模板
    print(1, 2, 3, "hello", 4.5);  // 可变参数

    return 0;
}
