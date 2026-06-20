// Stage 1 实战: 5 个 Python vs C++ 差异演示
// 编译: clang++ -std=c++17 01_basics.cpp -o 01_basics
// 运行: ./01_basics

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

// ===== 差异 1: 静态类型 =====
// Python 里 x 可以一会儿是 int 一会儿是 str
// C++ 里 x 一旦声明为 int，就永远是 int

void demo_static_typing() {
    std::cout << "\n=== 1. 静态类型 ===\n";
    int x = 42;
    std::cout << "x = " << x << " (int)\n";

    // 下面这行如果取消注释，编译会报错
    // x = "hello";  // error: cannot initialize a value of type 'int' with an lvalue of type 'const char[6]'

    // 隐式转换也不行
    // x = 3.14;  // warning: implicit conversion turns floating-point into int

    // C++ 必须显式转
    double pi = 3.14;
    int pi_int = static_cast<int>(pi);  // 显式转换
    std::cout << "pi 强转 int = " << pi_int << " (小数部分被砍掉)\n";
}

// ===== 差异 2: 编译执行 =====
// 不用 demo，看 hello.cpp 的汇编就知道，C++ 直接跑机器码
// 跑 ./01_basics 比 python 01_basics.py 快得多

void demo_compiled() {
    std::cout << "\n=== 2. 编译执行 ===\n";
    // 这段代码在编译时就被翻译成机器码
    // 运行时 CPU 直接执行，不经过任何解释器
    long sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    std::cout << "1+2+...+999999 = " << sum << " (CPU 直接跑，无解释器开销)\n";
}

// ===== 差异 3: 内存管理 =====
// Python: 列表超出作用域自动 GC
// C++: 必须用 RAII（vector 帮你管）

void demo_memory() {
    std::cout << "\n=== 3. 内存管理 ===\n";
    // C++ 推荐写法: 用 std::vector，自动管理
    {
        std::vector<int> v(1000, 0);  // 分配 1000 个 int
        v[0] = 1;
        std::cout << "v[0] = " << v[0] << " (vector 自动管理内存)\n";
    }  // 离开作用域，自动释放 v

    // ❌ 反面例子 (不要这么写):
    // int* p = new int[1000];
    // // ... 忘 delete 就泄漏了
    // delete[] p;
}

// ===== 差异 4: 值语义 =====
// Python: b = a 后改 b，a 也变
// C++: b = a 是拷贝，改 b 不影响 a

void demo_value_semantics() {
    std::cout << "\n=== 4. 值语义 ===\n";
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = a;  // b 是 a 的副本，不是引用

    b.push_back(4);

    std::cout << "a = ";
    for (int x : a) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "b = ";
    for (int x : b) std::cout << x << " ";
    std::cout << "\n";

    // 引用写法: 用 & 显式表达
    std::vector<int>& ref = a;  // ref 是 a 的引用（别名）
    ref.push_back(99);
    std::cout << "a 改了 ref 后: ";
    for (int x : a) std::cout << x << " ";
    std::cout << " (引用才会影响原对象)\n";
}

// ===== 差异 5: 错误处理 =====
// Python: 运行时报错
// C++: 编译期就抓

int safe_divide(int a, int b) {
    if (b == 0) {
        throw std::runtime_error("除数不能为 0");
    }
    return a / b;
}

void demo_error_handling() {
    std::cout << "\n=== 5. 错误处理 ===\n";
    try {
        int result = safe_divide(10, 0);
        std::cout << "10 / 0 = " << result << "\n";
    } catch (const std::exception& e) {
        std::cerr << "抓到异常: " << e.what() << "\n";
    }
    std::cout << "程序继续跑 (异常被 catch 住了)\n";
}

int main() {
    std::cout << "Stage 1: C++ vs Python 本质差异\n";
    std::cout << "================================\n";

    demo_static_typing();
    demo_compiled();
    demo_memory();
    demo_value_semantics();
    demo_error_handling();

    std::cout << "\n✅ 5 个差异都演示完了\n";
    return 0;
}
