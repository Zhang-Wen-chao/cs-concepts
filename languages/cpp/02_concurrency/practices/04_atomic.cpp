// 原子操作实践：无锁同步
// 编译：g++ -std=c++17 -pthread 04_atomic.cpp -o atomic_demo
// 运行：./atomic_demo
//
// 目的：演示原子操作的使用和性能对比

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>

// 演示 1：数据竞争 vs 原子操作
void demo_race_vs_atomic() {
    std::cout << "\n=== 演示 1：数据竞争 vs 原子操作 ===\n";

    // ❌ 非原子：数据竞争
    int counter1 = 0;
    std::thread t1([&]{ for(int i=0; i<100000; ++i) counter1++; });
    std::thread t2([&]{ for(int i=0; i<100000; ++i) counter1++; });
    t1.join();
    t2.join();
    std::cout << "非原子结果: " << counter1 << " (期望 200000)\n";

    // ✅ 原子：线程安全
    std::atomic<int> counter2(0);
    std::thread t3([&]{ for(int i=0; i<100000; ++i) counter2++; });
    std::thread t4([&]{ for(int i=0; i<100000; ++i) counter2++; });
    t3.join();
    t4.join();
    std::cout << "原子结果: " << counter2 << " (期望 200000)\n";
}

// 演示 2：原子操作的常用方法
void demo_atomic_operations() {
    std::cout << "\n=== 演示 2：原子操作方法 ===\n";

    std::atomic<int> a(0);

    // load/store
    a.store(10);
    std::cout << "load: " << a.load() << "\n";

    // 自增/自减
    a++;
    std::cout << "自增后: " << a << "\n";

    // fetch_add：返回旧值
    int old = a.fetch_add(5);
    std::cout << "fetch_add(5) 返回旧值: " << old << ", 新值: " << a << "\n";

    // exchange：设置新值，返回旧值
    old = a.exchange(100);
    std::cout << "exchange(100) 返回旧值: " << old << ", 新值: " << a << "\n";

    // compare_exchange_strong：CAS
    int expected = 100;
    int desired = 200;
    bool success = a.compare_exchange_strong(expected, desired);
    std::cout << "CAS 成功: " << success << ", 新值: " << a << "\n";

    // CAS 失败
    expected = 50;  // 错误的期望值
    success = a.compare_exchange_strong(expected, 300);
    std::cout << "CAS 失败: " << !success << ", expected 被更新为: " << expected << "\n";
}

// 演示 3：原子 bool（标志位）
void demo_atomic_flag() {
    std::cout << "\n=== 演示 3：原子 bool 标志位 ===\n";

    std::atomic<bool> ready(false);
    int data = 0;

    // 生产者
    std::thread producer([&]{
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        data = 42;
        ready.store(true, std::memory_order_release);
        std::cout << "生产者：数据已准备\n";
    });

    // 消费者
    std::thread consumer([&]{
        std::cout << "消费者：等待数据...\n";
        while (!ready.load(std::memory_order_acquire)) {
            // 忙等（实际应用中不推荐）
        }
        std::cout << "消费者：收到数据 " << data << "\n";
    });

    producer.join();
    consumer.join();
}

// 演示 4：自旋锁
class SpinLock {
    std::atomic<bool> flag_{false};
public:
    void lock() {
        while (flag_.exchange(true, std::memory_order_acquire)) {
            // 自旋等待
        }
    }

    void unlock() {
        flag_.store(false, std::memory_order_release);
    }
};

void demo_spinlock() {
    std::cout << "\n=== 演示 4：自旋锁 ===\n";

    SpinLock spinlock;
    int counter = 0;

    auto worker = [&]{
        for (int i = 0; i < 10000; ++i) {
            spinlock.lock();
            counter++;
            spinlock.unlock();
        }
    };

    std::thread t1(worker);
    std::thread t2(worker);

    t1.join();
    t2.join();

    std::cout << "自旋锁保护的计数器: " << counter << " (期望 20000)\n";
}

// 演示 5：性能对比（原子 vs 锁）
void demo_performance() {
    std::cout << "\n=== 演示 5：性能对比 ===\n";

    const int ITERATIONS = 1000000;

    // 测试原子操作
    {
        std::atomic<int> counter(0);
        auto start = std::chrono::steady_clock::now();

        std::thread t1([&]{ for(int i=0; i<ITERATIONS; ++i) counter++; });
        std::thread t2([&]{ for(int i=0; i<ITERATIONS; ++i) counter++; });

        t1.join();
        t2.join();

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "原子操作: " << elapsed.count() << " ms, 结果: " << counter << "\n";
    }

    // 测试锁
    {
        std::mutex mtx;
        int counter = 0;
        auto start = std::chrono::steady_clock::now();

        std::thread t1([&]{
            for(int i=0; i<ITERATIONS; ++i) {
                std::lock_guard<std::mutex> lock(mtx);
                counter++;
            }
        });
        std::thread t2([&]{
            for(int i=0; i<ITERATIONS; ++i) {
                std::lock_guard<std::mutex> lock(mtx);
                counter++;
            }
        });

        t1.join();
        t2.join();

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "互斥锁:   " << elapsed.count() << " ms, 结果: " << counter << "\n";
    }

    std::cout << "结论：原子操作比锁快（简单操作）\n";
}

// 演示 6：compare_exchange 实战
void demo_cas_usage() {
    std::cout << "\n=== 演示 6：compare_exchange 实战 ===\n";

    std::atomic<int> value(0);

    // 多线程尝试设置值
    auto worker = [&](int id, int target) {
        int expected = 0;
        if (value.compare_exchange_strong(expected, target)) {
            std::cout << "线程 " << id << " 成功设置为 " << target << "\n";
        } else {
            std::cout << "线程 " << id << " 失败，当前值: " << expected << "\n";
        }
    };

    std::thread t1(worker, 1, 100);
    std::thread t2(worker, 2, 200);
    std::thread t3(worker, 3, 300);

    t1.join();
    t2.join();
    t3.join();

    std::cout << "最终值: " << value << "\n";
}

// 演示 7：无锁累加器（CAS 循环）
void demo_lock_free_add() {
    std::cout << "\n=== 演示 7：无锁累加器 ===\n";

    std::atomic<int> counter(0);

    // 用 CAS 实现累加（实际应该用 fetch_add）
    auto add_with_cas = [](std::atomic<int>& target, int value) {
        int expected = target.load();
        while (!target.compare_exchange_weak(expected, expected + value)) {
            // CAS 失败，重试
        }
    };

    std::thread t1([&]{
        for (int i = 0; i < 10000; ++i) {
            add_with_cas(counter, 1);
        }
    });
    std::thread t2([&]{
        for (int i = 0; i < 10000; ++i) {
            add_with_cas(counter, 1);
        }
    });

    t1.join();
    t2.join();

    std::cout << "CAS 累加结果: " << counter << " (期望 20000)\n";
    std::cout << "注意：实际应该直接用 fetch_add\n";
}

int main() {
    std::cout << "原子操作演示\n";
    std::cout << "============\n";

    demo_race_vs_atomic();
    demo_atomic_operations();
    demo_atomic_flag();
    demo_spinlock();
    demo_performance();
    demo_cas_usage();
    demo_lock_free_add();

    std::cout << "\n所有演示完成！\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. 原子操作：不可分割的操作，无需锁
 *    std::atomic<int> a(0);
 *
 * 2. 常用方法：
 *    a.load()           // 读取
 *    a.store(10)        // 写入
 *    a++                // 自增
 *    a.fetch_add(5)     // 加 5，返回旧值
 *    a.exchange(100)    // 交换，返回旧值
 *    a.compare_exchange_strong(expected, desired)  // CAS
 *
 * 3. 性能：
 *    原子操作 > 锁（简单操作）
 *
 * 4. 适用场景：
 *    - 简单计数器
 *    - 标志位
 *    - 简单状态
 *
 * 5. 不适用场景：
 *    - 保护多个变量
 *    - 复杂操作
 *
 * 6. 内存顺序：
 *    - 默认：memory_order_seq_cst（最安全）
 *    - 优化：memory_order_acquire/release
 *    - 简单计数：memory_order_relaxed
 *
 * 7. CAS（compare_exchange）：
 *    - 最强大的原子操作
 *    - 用于无锁数据结构
 *    - weak 版本可能虚假失败，用于循环
 *    - strong 版本不会虚假失败
 */
