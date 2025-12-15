// 移动语义核心示例
// 编译：g++ -std=c++17 04_move_semantics.cpp -o move

#include <iostream>
#include <vector>
#include <string>

class MyVector {
    int* data_;
    size_t size_;
public:
    MyVector(size_t s) : size_(s), data_(new int[s]) {
        std::cout << "构造\n";
    }
    ~MyVector() { delete[] data_; std::cout << "析构\n"; }

    // 拷贝构造
    MyVector(const MyVector& o) : size_(o.size_), data_(new int[size_]) {
        std::copy(o.data_, o.data_ + size_, data_);
        std::cout << "拷贝构造\n";
    }

    // 移动构造
    MyVector(MyVector&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
        std::cout << "移动构造\n";
    }

    // 移动赋值
    MyVector& operator=(MyVector&& o) noexcept {
        if (this != &o) {
            delete[] data_;         // 释放自己的资源
            data_ = o.data_;        // 偷走 o 的资源
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
            std::cout << "移动赋值\n";
        }
        return *this;
    }
};

std::vector<int> create_vector() {
    return std::vector<int>(1000);  // 自动移动
}

int main() {
    // 1. 移动构造 vs 拷贝构造
    MyVector v1(100);
    MyVector v2 = v1;               // 拷贝构造
    MyVector v3 = std::move(v1);    // 移动构造

    // 2. 移动赋值
    MyVector v4(50);
    v4 = std::move(v2);             // 移动赋值（v4 已存在）

    // 3. 返回值自动移动
    auto vec = create_vector();     // 不拷贝

    // 4. 容器中移动
    std::string s1 = "hello";
    std::vector<std::string> v;
    v.push_back(s1);                // 拷贝
    v.push_back(std::move(s1));     // 移动（s1 变空）

    return 0;
}
