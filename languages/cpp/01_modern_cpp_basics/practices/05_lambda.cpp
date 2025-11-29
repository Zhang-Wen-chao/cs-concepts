/**
 * Lambda 表达式实践示例
 * 编译：g++ -std=c++17 05_lambda.cpp -o lambda
 * 运行：./lambda
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <string>
#include <numeric>

// ============ 示例 1：基本语法 ============

void test_basic_syntax() {
    std::cout << "\n=== 示例 1: Lambda 基本语法 ===" << std::endl;

    // 最简单的 Lambda
    auto hello = []() { std::cout << "Hello Lambda!" << std::endl; };
    hello();

    // 带参数
    auto add = [](int a, int b) { return a + b; };
    std::cout << "3 + 4 = " << add(3, 4) << std::endl;

    // 显式返回类型
    auto divide = [](int a, int b) -> double {
        return static_cast<double>(a) / b;
    };
    std::cout << "7 / 2 = " << divide(7, 2) << std::endl;

    // 立即调用
    int result = [](int x) { return x * 2; }(21);
    std::cout << "立即调用结果: " << result << std::endl;
}

// ============ 示例 2：捕获外部变量 ============

void test_capture() {
    std::cout << "\n=== 示例 2: 捕获外部变量 ===" << std::endl;

    int x = 10, y = 20;

    // []：不捕获
    auto f1 = []() {
        std::cout << "不捕获任何变量" << std::endl;
    };
    f1();

    // [x]：按值捕获 x
    auto f2 = [x]() {
        std::cout << "按值捕获 x = " << x << std::endl;
    };
    f2();

    // [&x]：按引用捕获 x
    auto f3 = [&x]() {
        x = 100;
        std::cout << "按引用捕获，修改 x = " << x << std::endl;
    };
    f3();
    std::cout << "外部 x = " << x << std::endl;

    // [x, &y]：x 按值，y 按引用
    x = 10;  // 重置
    auto f4 = [x, &y]() {
        std::cout << "x 按值 = " << x << ", y 按引用 = " << y << std::endl;
        y = 200;
    };
    f4();
    std::cout << "外部 y = " << y << std::endl;

    // [=]：按值捕获所有
    auto f5 = [=]() {
        std::cout << "按值捕获所有: x = " << x << ", y = " << y << std::endl;
    };
    f5();

    // [&]：按引用捕获所有
    auto f6 = [&]() {
        x = 30;
        y = 40;
        std::cout << "按引用捕获所有，修改后: x = " << x << ", y = " << y << std::endl;
    };
    f6();
    std::cout << "外部 x = " << x << ", y = " << y << std::endl;
}

// ============ 示例 3：按值 vs 按引用 ============

void test_value_vs_reference() {
    std::cout << "\n=== 示例 3: 按值 vs 按引用 ===" << std::endl;

    int counter = 0;

    // 按值捕获：捕获时拷贝
    auto f1 = [counter]() {
        std::cout << "按值捕获: " << counter << std::endl;
    };

    counter = 10;
    f1();  // 输出：0（捕获时的值）

    // 按引用捕获：使用当前值
    auto f2 = [&counter]() {
        std::cout << "按引用捕获: " << counter << std::endl;
    };

    counter = 20;
    f2();  // 输出：20（当前值）

    std::cout << "\n结论：" << std::endl;
    std::cout << "- 按值捕获：捕获时拷贝，后续修改不影响 Lambda" << std::endl;
    std::cout << "- 按引用捕获：使用原变量，修改会反映到 Lambda" << std::endl;
}

// ============ 示例 4：mutable 关键字 ============

void test_mutable() {
    std::cout << "\n=== 示例 4: mutable 关键字 ===" << std::endl;

    int x = 10;

    // 按值捕获默认是 const
    auto f1 = [x]() {
        // x = 20;  // ❌ 编译错误：不能修改
        std::cout << "const Lambda: x = " << x << std::endl;
    };
    f1();

    // mutable：可以修改捕获的值（但不影响原变量）
    auto f2 = [x]() mutable {
        x = 20;
        std::cout << "mutable Lambda: x = " << x << std::endl;
    };
    f2();
    f2();  // 再次调用
    std::cout << "外部 x = " << x << std::endl;  // 输出：10（没变）

    std::cout << "\n注意：mutable 修改的是 Lambda 内部的拷贝" << std::endl;
}

// ============ 示例 5：Lambda 与算法 ============

void test_with_algorithms() {
    std::cout << "\n=== 示例 5: Lambda 与标准算法 ===" << std::endl;

    std::vector<int> vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};

    // std::for_each：遍历
    std::cout << "\n原始数据: ";
    std::for_each(vec.begin(), vec.end(),
                  [](int x) { std::cout << x << " "; });
    std::cout << std::endl;

    // std::sort：排序
    std::sort(vec.begin(), vec.end(),
              [](int a, int b) { return a > b; });  // 降序
    std::cout << "降序排序: ";
    for (int x : vec) std::cout << x << " ";
    std::cout << std::endl;

    // std::find_if：查找
    auto it = std::find_if(vec.begin(), vec.end(),
                           [](int x) { return x < 5; });
    if (it != vec.end()) {
        std::cout << "第一个小于 5 的数: " << *it << std::endl;
    }

    // std::count_if：统计
    int count = std::count_if(vec.begin(), vec.end(),
                              [](int x) { return x % 2 == 0; });
    std::cout << "偶数个数: " << count << std::endl;

    // std::transform：转换
    std::vector<int> doubled(vec.size());
    std::transform(vec.begin(), vec.end(), doubled.begin(),
                   [](int x) { return x * 2; });
    std::cout << "每个元素乘以 2: ";
    for (int x : doubled) std::cout << x << " ";
    std::cout << std::endl;

    // std::accumulate：累加
    int sum = std::accumulate(vec.begin(), vec.end(), 0,
                              [](int acc, int x) { return acc + x; });
    std::cout << "总和: " << sum << std::endl;
}

// ============ 示例 6：Lambda 捕获实际应用 ============

void test_capture_use_case() {
    std::cout << "\n=== 示例 6: Lambda 捕获的实际应用 ===" << std::endl;

    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 统计大于阈值的数
    int threshold = 5;
    int count = std::count_if(vec.begin(), vec.end(),
                              [threshold](int x) { return x > threshold; });
    std::cout << "大于 " << threshold << " 的数: " << count << std::endl;

    // 修改阈值
    threshold = 7;
    count = std::count_if(vec.begin(), vec.end(),
                         [threshold](int x) { return x > threshold; });
    std::cout << "大于 " << threshold << " 的数: " << count << std::endl;

    // 累加器
    int total = 0;
    std::for_each(vec.begin(), vec.end(),
                  [&total](int x) { total += x; });
    std::cout << "总和: " << total << std::endl;
}

// ============ 示例 7：Lambda 的类型 ============

void test_lambda_type() {
    std::cout << "\n=== 示例 7: Lambda 的类型 ===" << std::endl;

    // 每个 Lambda 有唯一的类型
    auto f1 = [](int x) { return x + 1; };
    auto f2 = [](int x) { return x + 1; };

    std::cout << "f1(10) = " << f1(10) << std::endl;
    std::cout << "f2(10) = " << f2(10) << std::endl;

    // f1 和 f2 类型不同，不能赋值
    // f1 = f2;  // ❌ 编译错误

    // 用 std::function 存储
    std::function<int(int)> func1 = f1;
    std::function<int(int)> func2 = f2;

    std::cout << "\n用 std::function 存储后可以统一类型" << std::endl;

    // 可以放到容器中
    std::vector<std::function<int(int)>> funcs = {func1, func2};
    std::cout << "容器中有 " << funcs.size() << " 个函数" << std::endl;
}

// ============ 示例 8：泛型 Lambda（C++14）============

void test_generic_lambda() {
    std::cout << "\n=== 示例 8: 泛型 Lambda（C++14）===" << std::endl;

    // 参数类型用 auto
    auto print = [](const auto& x) {
        std::cout << "值: " << x << std::endl;
    };

    print(42);           // int
    print(3.14);         // double
    print("hello");      // const char*
    print(std::string("world"));  // string

    // 泛型 Lambda 配合算法
    std::vector<int> int_vec = {1, 2, 3};
    std::vector<double> double_vec = {1.1, 2.2, 3.3};

    auto sum = [](const auto& vec) {
        return std::accumulate(vec.begin(), vec.end(), 0.0);
    };

    std::cout << "int_vec 总和: " << sum(int_vec) << std::endl;
    std::cout << "double_vec 总和: " << sum(double_vec) << std::endl;
}

// ============ 示例 9：常见陷阱 - 悬空引用 ============

std::function<int()> create_bad_lambda() {
    int x = 10;
    return [&x]() { return x; };  // ❌ 危险：x 的生命周期结束
}

std::function<int()> create_good_lambda() {
    int x = 10;
    return [x]() { return x; };  // ✅ 安全：拷贝 x
}

void test_dangling_reference() {
    std::cout << "\n=== 示例 9: 常见陷阱 - 悬空引用 ===" << std::endl;

    std::cout << "\n❌ 错误做法（已注释，避免未定义行为）：" << std::endl;
    std::cout << "// auto bad = create_bad_lambda();" << std::endl;
    std::cout << "// int result = bad();  // 未定义行为" << std::endl;

    std::cout << "\n✅ 正确做法：" << std::endl;
    auto good = create_good_lambda();
    int result = good();
    std::cout << "结果: " << result << std::endl;

    std::cout << "\n结论：返回 Lambda 时，不要按引用捕获局部变量" << std::endl;
}

// ============ 示例 10：按值捕获大对象 ============

void test_capture_large_object() {
    std::cout << "\n=== 示例 10: 按值捕获大对象 ===" << std::endl;

    std::vector<int> large_vec(1000000, 42);

    std::cout << "large_vec 大小: " << large_vec.size() << std::endl;

    // ❌ 低效：拷贝整个 vector
    auto f1 = [large_vec]() {
        return large_vec.size();
    };
    std::cout << "按值捕获（拷贝）: size = " << f1() << std::endl;

    // ✅ 高效：按引用捕获
    auto f2 = [&large_vec]() {
        return large_vec.size();
    };
    std::cout << "按引用捕获: size = " << f2() << std::endl;

    // ✅ 更好：只捕获需要的
    auto size = large_vec.size();
    auto f3 = [size]() {
        return size;
    };
    std::cout << "只捕获 size: " << f3() << std::endl;

    std::cout << "\n建议：" << std::endl;
    std::cout << "- 小对象：按值捕获" << std::endl;
    std::cout << "- 大对象：按引用捕获或只捕获需要的成员" << std::endl;
}

// ============ 示例 11：初始化捕获（C++14）============

void test_init_capture() {
    std::cout << "\n=== 示例 11: 初始化捕获（C++14）===" << std::endl;

    // 移动捕获 unique_ptr
    auto ptr = std::make_unique<int>(42);
    std::cout << "创建 unique_ptr，值: " << *ptr << std::endl;

    auto lambda = [p = std::move(ptr)]() {
        std::cout << "Lambda 中访问: " << *p << std::endl;
        return *p;
    };

    // ptr 现在是空的
    if (!ptr) {
        std::cout << "ptr 已被移动，现在是空的" << std::endl;
    }

    lambda();

    // 自定义初始化
    auto lambda2 = [x = 10, y = 20]() {
        return x + y;
    };
    std::cout << "自定义初始化: " << lambda2() << std::endl;
}

// ============ 示例 12：立即调用的 Lambda（IIFE）============

void test_iife() {
    std::cout << "\n=== 示例 12: 立即调用的 Lambda（IIFE）===" << std::endl;

    // 复杂的初始化
    bool condition = true;
    int x = [condition]() {
        if (condition) {
            std::cout << "条件为真，返回 42" << std::endl;
            return 42;
        } else {
            std::cout << "条件为假，返回 100" << std::endl;
            return 100;
        }
    }();  // 立即调用

    std::cout << "x = " << x << std::endl;

    // 用于 const 初始化
    const std::string message = [&x]() {
        if (x > 50) {
            return std::string("大数");
        } else {
            return std::string("小数");
        }
    }();

    std::cout << "message = " << message << std::endl;
}

// ============ 示例 13：实际应用 - 自定义排序 ============

struct Person {
    std::string name;
    int age;
};

void test_real_world_example() {
    std::cout << "\n=== 示例 13: 实际应用 - 自定义排序 ===" << std::endl;

    std::vector<Person> people = {
        {"Alice", 30},
        {"Bob", 25},
        {"Charlie", 35},
        {"David", 25}
    };

    std::cout << "\n原始顺序:" << std::endl;
    for (const auto& p : people) {
        std::cout << "  " << p.name << ", " << p.age << " 岁" << std::endl;
    }

    // 按年龄排序
    std::sort(people.begin(), people.end(),
              [](const Person& a, const Person& b) {
                  return a.age < b.age;
              });

    std::cout << "\n按年龄排序:" << std::endl;
    for (const auto& p : people) {
        std::cout << "  " << p.name << ", " << p.age << " 岁" << std::endl;
    }

    // 按名字长度排序
    std::sort(people.begin(), people.end(),
              [](const Person& a, const Person& b) {
                  return a.name.length() < b.name.length();
              });

    std::cout << "\n按名字长度排序:" << std::endl;
    for (const auto& p : people) {
        std::cout << "  " << p.name << ", " << p.age << " 岁" << std::endl;
    }
}

// ============ 示例 14：Lambda 最佳实践 ============

void print_best_practices() {
    std::cout << "\n=== 示例 14: Lambda 最佳实践 ===" << std::endl;

    std::cout << "\n1. 默认用 auto 存储 Lambda" << std::endl;
    std::cout << "   ✅ auto f = [](int x) { return x + 1; };" << std::endl;
    std::cout << "   ⚠️  std::function<int(int)> f = ...;  // 有开销" << std::endl;

    std::cout << "\n2. 小心按引用捕获" << std::endl;
    std::cout << "   ✅ 立即使用：可以按引用" << std::endl;
    std::cout << "   ❌ 延迟使用：按值捕获更安全" << std::endl;

    std::cout << "\n3. 捕获建议" << std::endl;
    std::cout << "   - 小对象：按值 [x]" << std::endl;
    std::cout << "   - 大对象：按引用 [&x] 或只捕获需要的成员" << std::endl;
    std::cout << "   - 移动语义：[p = std::move(ptr)]" << std::endl;

    std::cout << "\n4. 返回 Lambda" << std::endl;
    std::cout << "   ❌ 不要按引用捕获局部变量" << std::endl;
    std::cout << "   ✅ 按值捕获或初始化捕获" << std::endl;

    std::cout << "\n5. 配合算法" << std::endl;
    std::cout << "   - Lambda 让算法更简洁、更灵活" << std::endl;
    std::cout << "   - 优先用标准算法 + Lambda，不要手写循环" << std::endl;
}

// ============ 主函数 ============

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      Lambda 表达式实践示例" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_basic_syntax();
        test_capture();
        test_value_vs_reference();
        test_mutable();
        test_with_algorithms();
        test_capture_use_case();
        test_lambda_type();
        test_generic_lambda();
        test_dangling_reference();
        test_capture_large_object();
        test_init_capture();
        test_iife();
        test_real_world_example();
        print_best_practices();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  所有示例运行完成！✅" << std::endl;
        std::cout << "========================================" << std::endl;

        std::cout << "\n关键收获：" << std::endl;
        std::cout << "1. Lambda = 就地定义的匿名函数" << std::endl;
        std::cout << "2. [x] 按值捕获，[&x] 按引用捕获" << std::endl;
        std::cout << "3. 配合算法使用，代码更简洁" << std::endl;
        std::cout << "4. 默认用 auto 存储（零开销）" << std::endl;
        std::cout << "5. 小心悬空引用（返回 Lambda 时）" << std::endl;
        std::cout << "6. mutable 修改的是 Lambda 内部的拷贝" << std::endl;
        std::cout << "7. 泛型 Lambda 用 auto 参数（C++14）" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
