// 旧 vs 新 C++ 对比
// 编译：g++ -std=c++17 00_old_vs_new_cpp.cpp -o compare

#include <iostream>
#include <vector>
#include <memory>
#include <string>

void old_style() {
    // ❌ 旧：手动管理
    int* arr = new int[100];
    // ... 使用 ...
    delete[] arr;  // 容易忘记
}

void new_style() {
    // ✅ 新：自动管理
    std::vector<int> arr(100);
    // 自动释放
}

void old_pointer() {
    // ❌ 旧：裸指针
    int* p = new int(42);
    delete p;
}

void new_pointer() {
    // ✅ 新：智能指针
    auto p = std::make_unique<int>(42);
}

void old_string() {
    // ❌ 旧：C 字符串
    char* str = new char[100];
    strcpy(str, "hello");
    delete[] str;
}

void new_string() {
    // ✅ 新：std::string
    std::string str = "hello";
}

int main() {
    std::cout << "旧 C++ vs 现代 C++\n";
    new_style();
    new_pointer();
    new_string();
    return 0;
}
