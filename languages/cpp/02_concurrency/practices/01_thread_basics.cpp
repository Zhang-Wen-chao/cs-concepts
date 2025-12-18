// 线程基础实践：多线程并行求和
// 编译：g++ -std=c++17 -pthread 01_thread_basics.cpp -o demo
// 运行：./demo
//
// 目的：演示如何创建多个线程并行处理数据

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

// 演示 1：基本线程创建
void demo_basic() {
    std::cout << "\n=== 演示 1：基本线程创建 ===\n";

    // 方式 1：Lambda
    std::thread t1([]{
        std::cout << "线程 1 (lambda) 运行中，ID: "
                  << std::this_thread::get_id() << "\n";
    });

    // 方式 2：函数
    auto work = []{
        std::cout << "线程 2 (函数) 运行中，ID: "
                  << std::this_thread::get_id() << "\n";
    };
    std::thread t2(work);

    // 必须 join（等待线程结束）
    t1.join();
    t2.join();
}

// 演示 2：传递参数
void demo_params() {
    std::cout << "\n=== 演示 2：传递参数 ===\n";

    // 按值传递
    std::thread t1([](int x, std::string s){
        std::cout << "参数: " << x << ", " << s << "\n";
    }, 42, "hello");

    // 按引用传递（用 std::ref）
    int count = 0;
    std::thread t2([](int& c){
        c = 100;
    }, std::ref(count));

    t1.join();
    t2.join();

    std::cout << "修改后的 count: " << count << "\n";  // 100
}

// 演示 3：多线程并行求和
void demo_parallel_sum() {
    std::cout << "\n=== 演示 3：多线程并行求和 ===\n";

    // 1. 准备数据：1 到 1000000
    const int N = 1000000;
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i) {
        data[i] = i + 1;
    }

    // 2. 获取 CPU 核心数
    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    std::cout << "使用 " << num_threads << " 个线程\n";

    // 3. 每个线程计算部分和
    std::vector<std::thread> threads;
    std::vector<long long> partial_sums(num_threads, 0);

    int chunk_size = N / num_threads;

    auto start = std::chrono::steady_clock::now();

    for (unsigned i = 0; i < num_threads; ++i) {
        int begin = i * chunk_size;
        int end = (i == num_threads - 1) ? N : (i + 1) * chunk_size;

        threads.emplace_back([&data, &partial_sums, i, begin, end]{
            long long sum = 0;
            for (int j = begin; j < end; ++j) {
                sum += data[j];
            }
            partial_sums[i] = sum;

            std::cout << "线程 " << i << " 处理 [" << begin << ", " << end
                      << ")，部分和: " << sum << "\n";
        });
    }

    // 4. 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 5. 汇总结果
    long long total = 0;
    for (long long sum : partial_sums) {
        total += sum;
    }

    // 验证结果（等差数列求和公式）
    long long expected = (long long)N * (N + 1) / 2;

    std::cout << "\n总和: " << total << "\n";
    std::cout << "预期: " << expected << "\n";
    std::cout << "正确: " << (total == expected ? "✓" : "✗") << "\n";
    std::cout << "耗时: " << elapsed.count() << " ms\n";
}

// 演示 4：RAII 线程管理
class ThreadGuard {
    std::thread& t_;
public:
    explicit ThreadGuard(std::thread& t) : t_(t) {}
    ~ThreadGuard() {
        if (t_.joinable()) t_.join();
    }

    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};

void demo_raii() {
    std::cout << "\n=== 演示 4：RAII 线程管理 ===\n";

    std::thread t([]{
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "线程完成\n";
    });

    ThreadGuard guard(t);  // 析构时自动 join

    std::cout << "即使有异常或提前返回，线程也会被正确回收\n";
    // 离开作用域，guard 析构 → 自动 join
}

int main() {
    std::cout << "线程基础演示\n";
    std::cout << "=============\n";

    demo_basic();
    demo_params();
    demo_parallel_sum();
    demo_raii();

    std::cout << "\n所有演示完成！\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. 线程创建：
 *    std::thread t(函数);
 *    std::thread t([]{...});  // Lambda
 *
 * 2. 必须 join 或 detach：
 *    t.join();    // 等待线程结束（推荐）
 *    t.detach();  // 分离线程（慎用）
 *
 * 3. 传递参数：
 *    std::thread t(func, arg1, arg2);        // 按值
 *    std::thread t(func, std::ref(var));     // 按引用
 *
 * 4. 获取 CPU 核心数：
 *    std::thread::hardware_concurrency();
 *
 * 5. RAII 管理：
 *    用 ThreadGuard 包装，避免忘记 join
 *
 * 6. 常见陷阱：
 *    - 忘记 join/detach → 程序崩溃
 *    - 引用捕获局部变量 + detach → 悬空引用
 *    - 线程数过多 → 性能下降
 */
