// Lambda 核心示例
// 编译：g++ -std=c++17 05_lambda.cpp -o lambda

#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 1. 基本 lambda
    auto print = [](int x) { std::cout << x << " "; };
    std::for_each(v.begin(), v.end(), print);
    std::cout << "\n";

    // 2. 排序
    std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });  // 降序

    // 3. 查找
    auto it = std::find_if(v.begin(), v.end(), [](int x) { return x > 5; });
    if (it != v.end()) std::cout << "Found: " << *it << "\n";

    // 4. 捕获
    int threshold = 5;
    auto count = std::count_if(v.begin(), v.end(),
                               [threshold](int x) { return x > threshold; });
    std::cout << "Count > " << threshold << ": " << count << "\n";

    // 5. 按值捕获所有
    int sum = 0;
    std::for_each(v.begin(), v.end(), [=, &sum](int x) { sum += x; });
    std::cout << "Sum: " << sum << "\n";

    // 6. 按引用捕获
    int multiplier = 2;
    std::for_each(v.begin(), v.end(), [&multiplier](int& x) { x *= multiplier; });

    return 0;
}
