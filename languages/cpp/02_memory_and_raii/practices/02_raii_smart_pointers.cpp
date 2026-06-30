// Stage 2 实战: RAII + 智能指针 + 移动语义
// 编译: clang++ -std=c++17 02_raii_smart_pointers.cpp -o 02_raii_smart_pointers
// 运行: ./02_raii_smart_pointers

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

// ===== 1. 栈 vs 堆：基本演示 =====
void demo_stack_vs_heap() {
    std::cout << "\n=== 1. 栈 vs 堆 ===\n";

    // 栈上：超快，自动释放
    int a = 42;
    int arr[100] = {0};
    std::cout << "栈变量 a 地址: " << &a << "\n";
    std::cout << "栈数组 arr 地址: " << &arr << "\n";

    // 堆上：手动管理
    int* p = new int(42);
    int* arr_p = new int[100];
    std::cout << "堆变量 p 地址: " << p << "\n";
    std::cout << "堆数组 arr_p 地址: " << arr_p << "\n";

    delete p;
    delete[] arr_p;
    std::cout << "✅ 堆内存手动释放\n";

    // 栈地址高，堆地址低（macOS 上）
    std::cout << "栈 vs 堆：栈地址更高，堆地址更低\n";
}

// ===== 2. 裸指针的 3 种典型 bug =====
void demo_raw_pointer_bugs() {
    std::cout << "\n=== 2. 裸指针的 bug（演示问题，不真触发崩溃）===\n";

    // Bug 1: 内存泄漏（new 后忘 delete）
    // 这里故意不 delete 演示
    auto leak_demo = []() {
        int* p = new int[1000];
        // 假设中间发生错误，return 跳过了 delete
        // delete[] p;  // ❌ 漏了
    };
    std::cout << "Bug 1: 内存泄漏 — 忘了 delete\n";

    // Bug 2: 悬空指针（delete 后还用）
    auto dangling_demo = []() {
        int* p = new int(42);
        delete p;
        // *p = 99;  // ❌ use-after-free，行为未定义
    };
    std::cout << "Bug 2: 悬空指针 — delete 后还用\n";

    // Bug 3: 双重释放
    auto double_free_demo = []() {
        int* p = new int(42);
        delete p;
        // delete p;  // ❌ double-free，程序崩
    };
    std::cout << "Bug 3: 双重释放 — delete 两次\n";
}

// ===== 3. unique_ptr：90% 场景的解 =====
void demo_unique_ptr() {
    std::cout << "\n=== 3. unique_ptr: 独占所有权 ===\n";

    // 创建（C++14+ make_unique）
    std::unique_ptr<int> p = std::make_unique<int>(42);
    std::cout << "*p = " << *p << "\n";

    // 跟裸指针一样用
    *p = 99;
    std::cout << "改后 *p = " << *p << "\n";

    // 不能拷贝
    // std::unique_ptr<int> p2 = p;  // ❌ 编译错误

    // 但能 move（转移所有权）
    std::unique_ptr<int> p2 = std::move(p);
    std::cout << "move 后 p 是否为空: " << (p ? "否" : "是") << "\n";
    std::cout << "move 后 *p2 = " << *p2 << "\n";

    // 离开作用域，p2 自动 delete
    std::cout << "✅ 离开作用域后自动释放\n";
}

// ===== 4. shared_ptr + weak_ptr：共享所有权 + 打破循环 =====
struct Node {
    std::string name;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // 用 weak 打破循环

    Node(const std::string& n) : name(n) {
        std::cout << "构造 " << name << "\n";
    }
    ~Node() {
        std::cout << "析构 " << name << "\n";
    }
};

void demo_shared_weak() {
    std::cout << "\n=== 4. shared_ptr + weak_ptr ===\n";
    {
        auto a = std::make_shared<Node>("A");
        auto b = std::make_shared<Node>("B");
        a->next = b;       // b 引用计数：1
        b->prev = a;       // a 引用计数还是 1（weak 不增加）
        std::cout << "a.use_count() = " << a.use_count() << "\n";
        std::cout << "b.use_count() = " << b.use_count() << "\n";
    }  // a、b 离开作用域，引用计数归 0，自动析构
    std::cout << "✅ 没有循环引用，正常析构\n";
}

// ===== 5. 移动语义 vs 拷贝：性能差 =====
struct BigBuffer {
    std::vector<int> data;

    BigBuffer(size_t n) : data(n, 0) {
        std::cout << "构造 " << n << " 个 int\n";
    }

    // 拷贝构造
    BigBuffer(const BigBuffer& other) : data(other.data) {
        std::cout << "拷贝构造（慢：复制 " << data.size() << " 个 int）\n";
    }

    // 移动构造
    BigBuffer(BigBuffer&& other) noexcept : data(std::move(other.data)) {
        std::cout << "移动构造（快：O(1) 转移指针）\n";
    }
};

void demo_move_vs_copy() {
    std::cout << "\n=== 5. 移动 vs 拷贝 ===\n";
    BigBuffer src(1000000);  // 100 万元素

    std::cout << "--- 拷贝 ---\n";
    auto start = std::chrono::high_resolution_clock::now();
    BigBuffer copy = src;  // 拷贝
    auto end = std::chrono::high_resolution_clock::now();
    auto copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "拷贝耗时: " << copy_ms << " ms\n";

    std::cout << "--- 移动 ---\n";
    start = std::chrono::high_resolution_clock::now();
    BigBuffer moved = std::move(src);  // 移动
    end = std::chrono::high_resolution_clock::now();
    auto move_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "移动耗时: " << move_us << " us （约 " << move_us / 1000.0 << " ms）\n";

    std::cout << "✅ 移动比拷贝快 100-1000 倍\n";
}

// ===== 6. 自己写 RAII 包装 =====
class Timer {
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {
        std::cout << "[start] " << name_ << "\n";
    }
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::cout << "[end]   " << name_ << " (耗时 " << ms << " us)\n";
    }
};

void demo_custom_raii() {
    std::cout << "\n=== 6. 自己写的 RAII: Timer ===\n";
    Timer t("demo_custom_raii");
    // ... 干很多事 ...
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // 离开作用域，t 自动析构，打印耗时
}

int main() {
    std::cout << "Stage 2: 内存模型 + RAII + 智能指针 + 移动语义\n";
    std::cout << "================================================\n";

    demo_stack_vs_heap();
    demo_raw_pointer_bugs();
    demo_unique_ptr();
    demo_shared_weak();
    demo_move_vs_copy();
    demo_custom_raii();

    std::cout << "\n✅ 6 个 demo 都跑完了\n";
    return 0;
}
