// 性能优化实践：常见优化技术
// 编译：g++ -std=c++17 -O2 04_performance_demo.cpp -o performance_demo
// 运行：./performance_demo
//
// 目的：演示各种性能优化技术及其效果

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <thread>
#include <atomic>

// RAII 计时器
class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    const char* name_;
public:
    Timer(const char* name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << ": " << duration.count() << " μs\n";
    }
};

// ===== 演示 1：减少拷贝 =====

// ❌ 拷贝
void process_copy(std::vector<int> vec) {
    int sum = 0;
    for (int x : vec) sum += x;
}

// ✅ 引用
void process_ref(const std::vector<int>& vec) {
    int sum = 0;
    for (int x : vec) sum += x;
}

void demo_copy_vs_ref() {
    std::cout << "===== 演示 1：减少拷贝 =====\n";
    std::vector<int> vec(1000000, 42);

    {
        Timer t("拷贝传递");
        for (int i = 0; i < 100; ++i) {
            process_copy(vec);
        }
    }

    {
        Timer t("引用传递");
        for (int i = 0; i < 100; ++i) {
            process_ref(vec);
        }
    }

    std::cout << "\n";
}

// ===== 演示 2：预分配 vs 动态扩容 =====

void demo_reserve() {
    std::cout << "===== 演示 2：预分配 vs 动态扩容 =====\n";
    const int N = 1000000;

    {
        Timer t("不预分配");
        std::vector<int> vec;
        for (int i = 0; i < N; ++i) {
            vec.push_back(i);  // 多次扩容
        }
    }

    {
        Timer t("预分配");
        std::vector<int> vec;
        vec.reserve(N);  // 一次分配
        for (int i = 0; i < N; ++i) {
            vec.push_back(i);
        }
    }

    std::cout << "\n";
}

// ===== 演示 3：push_back vs emplace_back =====

struct Point {
    int x, y;
    Point(int x, int y) : x(x), y(y) {}
};

void demo_emplace() {
    std::cout << "===== 演示 3：push_back vs emplace_back =====\n";
    const int N = 1000000;

    {
        Timer t("push_back");
        std::vector<Point> vec;
        vec.reserve(N);
        for (int i = 0; i < N; ++i) {
            vec.push_back(Point(i, i));  // 构造临时对象 → 移动
        }
    }

    {
        Timer t("emplace_back");
        std::vector<Point> vec;
        vec.reserve(N);
        for (int i = 0; i < N; ++i) {
            vec.emplace_back(i, i);  // 原地构造
        }
    }

    std::cout << "\n";
}

// ===== 演示 4：连续内存 vs 链表 =====

struct Node {
    int data;
    Node* next;
};

void demo_cache_friendly() {
    std::cout << "===== 演示 4：连续内存 vs 链表 =====\n";
    const int N = 100000;

    // 链表
    Node* head = nullptr;
    for (int i = 0; i < N; ++i) {
        Node* node = new Node{i, head};
        head = node;
    }

    {
        Timer t("链表遍历");
        long long sum = 0;
        for (Node* p = head; p; p = p->next) {
            sum += p->data;
        }
    }

    // vector
    std::vector<int> vec(N);
    std::iota(vec.begin(), vec.end(), 0);

    {
        Timer t("vector 遍历");
        long long sum = 0;
        for (int x : vec) {
            sum += x;
        }
    }

    // 清理链表
    while (head) {
        Node* tmp = head;
        head = head->next;
        delete tmp;
    }

    std::cout << "\n";
}

// ===== 演示 5：循环不变量提升 =====

int expensive_function() {
    int result = 0;
    for (int i = 0; i < 1000; ++i) {
        result += i;
    }
    return result;
}

void demo_loop_invariant() {
    std::cout << "===== 演示 5：循环不变量提升 =====\n";
    std::vector<int> vec(10000, 1);

    {
        Timer t("循环内计算");
        int result = 0;
        for (int i = 0; i < vec.size(); ++i) {
            result += vec[i] * expensive_function();  // 重复计算
        }
    }

    {
        Timer t("提升到循环外");
        int result = 0;
        int factor = expensive_function();  // 只计算一次
        for (int i = 0; i < vec.size(); ++i) {
            result += vec[i] * factor;
        }
    }

    std::cout << "\n";
}

// ===== 演示 6：选择合适的数据结构 =====

void demo_data_structure() {
    std::cout << "===== 演示 6：选择合适的数据结构 =====\n";
    const int N = 100000;

    // vector 查找
    std::vector<int> vec(N);
    std::iota(vec.begin(), vec.end(), 0);

    {
        Timer t("vector 查找 1000 次");
        for (int i = 0; i < 1000; ++i) {
            auto it = std::find(vec.begin(), vec.end(), N / 2);
        }
    }

    // unordered_set 查找
    std::unordered_set<int> set(vec.begin(), vec.end());

    {
        Timer t("unordered_set 查找 1000 次");
        for (int i = 0; i < 1000; ++i) {
            set.count(N / 2);
        }
    }

    std::cout << "\n";
}

// ===== 演示 7：多线程优化 =====

void compute_sum(const std::vector<int>& vec, size_t start, size_t end, std::atomic<long long>& result) {
    long long local_sum = 0;
    for (size_t i = start; i < end; ++i) {
        local_sum += vec[i];
    }
    result += local_sum;
}

void demo_multithread() {
    std::cout << "===== 演示 7：单线程 vs 多线程 =====\n";
    const int N = 10000000;
    std::vector<int> vec(N, 1);

    {
        Timer t("单线程");
        long long sum = 0;
        for (int x : vec) {
            sum += x;
        }
    }

    {
        Timer t("多线程（4 个）");
        std::atomic<long long> sum(0);
        unsigned num_threads = 4;
        std::vector<std::thread> threads;
        size_t chunk = vec.size() / num_threads;

        for (unsigned i = 0; i < num_threads; ++i) {
            size_t start = i * chunk;
            size_t end = (i == num_threads - 1) ? vec.size() : (i + 1) * chunk;
            threads.emplace_back(compute_sum, std::ref(vec), start, end, std::ref(sum));
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    std::cout << "\n";
}

// ===== 演示 8：减少不必要的操作 =====

void demo_unnecessary_work() {
    std::cout << "===== 演示 8：减少不必要的操作 =====\n";
    std::vector<int> vec;

    {
        Timer t("每次都排序");
        for (int i = 0; i < 1000; ++i) {
            vec.push_back(i);
            std::sort(vec.begin(), vec.end());  // 1000 次排序
        }
    }

    vec.clear();

    {
        Timer t("最后排序一次");
        for (int i = 0; i < 1000; ++i) {
            vec.push_back(i);
        }
        std::sort(vec.begin(), vec.end());  // 1 次
    }

    std::cout << "\n";
}

int main() {
    std::cout << "性能优化演示\n";
    std::cout << "==================\n\n";

    demo_copy_vs_ref();
    demo_reserve();
    demo_emplace();
    demo_cache_friendly();
    demo_loop_invariant();
    demo_data_structure();
    demo_multithread();
    demo_unnecessary_work();

    std::cout << "提示：用 -O2 或 -O3 编译以获得最佳性能\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. 减少拷贝：
 *    - 引用传递（const&）
 *    - 移动语义
 *    - emplace 原地构造
 *
 * 2. 减少分配：
 *    - reserve 预分配
 *    - 对象池/内存池
 *
 * 3. 缓存友好：
 *    - vector 比链表快 10-100 倍
 *    - 连续内存访问
 *
 * 4. 循环优化：
 *    - 提升不变量
 *    - 缓存 size
 *
 * 5. 数据结构：
 *    - 频繁查找用 unordered_set/map
 *    - 默认用 vector
 *
 * 6. 并发：
 *    - CPU 密集型用多线程
 *    - 注意线程开销
 *
 * 7. 减少不必要的操作：
 *    - 批量处理
 *    - 延迟计算
 *
 * 8. 测量：
 *    - 用 Timer 测量
 *    - 对比优化前后
 */
