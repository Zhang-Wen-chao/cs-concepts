// 互斥锁实践：演示数据竞争和锁的使用
// 编译：g++ -std=c++17 -pthread 02_mutex_locks.cpp -o mutex_locks
// 运行：./mutex_locks
//
// 目的：演示为什么需要锁，以及如何正确使用 RAII 锁

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>

// 演示 1：数据竞争（不加锁）
void demo_race_condition() {
    std::cout << "\n=== 演示 1：数据竞争 ===\n";

    int counter = 0;

    // 创建 2 个线程，各自累加 100000 次
    std::thread t1([&]{
        for (int i = 0; i < 100000; ++i) {
            counter++;  // 数据竞争
        }
    });

    std::thread t2([&]{
        for (int i = 0; i < 100000; ++i) {
            counter++;  // 数据竞争
        }
    });

    t1.join();
    t2.join();

    std::cout << "期望结果: 200000\n";
    std::cout << "实际结果: " << counter << "\n";
    std::cout << "正确: " << (counter == 200000 ? "✓" : "✗ 数据丢失") << "\n";
}

// 演示 2：使用 mutex（手动 lock/unlock）
void demo_manual_lock() {
    std::cout << "\n=== 演示 2：手动 lock/unlock（不推荐）===\n";

    std::mutex mtx;
    int counter = 0;

    std::thread t1([&]{
        for (int i = 0; i < 100000; ++i) {
            mtx.lock();
            counter++;
            mtx.unlock();
        }
    });

    std::thread t2([&]{
        for (int i = 0; i < 100000; ++i) {
            mtx.lock();
            counter++;
            mtx.unlock();
        }
    });

    t1.join();
    t2.join();

    std::cout << "结果: " << counter << "\n";
    std::cout << "正确: " << (counter == 200000 ? "✓" : "✗") << "\n";
    std::cout << "问题：手动 unlock 容易忘记，异常时也不会解锁\n";
}

// 演示 3：使用 lock_guard（推荐）
void demo_lock_guard() {
    std::cout << "\n=== 演示 3：lock_guard（推荐）===\n";

    std::mutex mtx;
    int counter = 0;

    std::thread t1([&]{
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard<std::mutex> lock(mtx);  // RAII 自动管理
            counter++;
            // 离开作用域自动解锁
        }
    });

    std::thread t2([&]{
        for (int i = 0; i < 100000; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            counter++;
        }
    });

    t1.join();
    t2.join();

    std::cout << "结果: " << counter << "\n";
    std::cout << "正确: " << (counter == 200000 ? "✓" : "✗") << "\n";
    std::cout << "优势：自动解锁，异常安全\n";
}

// 演示 4：unique_lock（更灵活）
void demo_unique_lock() {
    std::cout << "\n=== 演示 4：unique_lock（灵活控制）===\n";

    std::mutex mtx;
    int data = 0;

    auto worker = [&]{
        std::unique_lock<std::mutex> lock(mtx);  // 加锁
        data++;
        std::cout << "线程 " << std::this_thread::get_id()
                  << " 修改 data = " << data << "\n";

        lock.unlock();  // 手动解锁

        // 做一些不需要锁的工作
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        lock.lock();  // 再次加锁
        data++;
        std::cout << "线程 " << std::this_thread::get_id()
                  << " 再次修改 data = " << data << "\n";
        // 离开作用域自动解锁
    };

    std::thread t1(worker);
    std::thread t2(worker);

    t1.join();
    t2.join();

    std::cout << "最终 data = " << data << "\n";
}

// 演示 5：锁的粒度
void demo_lock_granularity() {
    std::cout << "\n=== 演示 5：锁的粒度 ===\n";

    std::mutex mtx;
    std::vector<int> shared_data;

    auto bad_worker = [&]{
        std::lock_guard<std::mutex> lock(mtx);
        // ❌ 坏：整个计算都在锁里
        int result = 0;
        for (int i = 0; i < 1000; ++i) {
            result += i;  // 耗时计算
        }
        shared_data.push_back(result);
    };

    auto good_worker = [&]{
        // ✅ 好：只在必要时加锁
        int result = 0;
        for (int i = 0; i < 1000; ++i) {
            result += i;  // 耗时计算（不需要锁）
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            shared_data.push_back(result);  // 只锁这一行
        }
    };

    std::cout << "好的做法：只在访问共享数据时加锁\n";
    std::cout << "坏的做法：把整个计算过程都锁住\n";

    std::thread t1(good_worker);
    std::thread t2(good_worker);
    t1.join();
    t2.join();

    std::cout << "shared_data 大小: " << shared_data.size() << "\n";
}

// 演示 6：scoped_lock（C++17，多个锁）
void demo_scoped_lock() {
    std::cout << "\n=== 演示 6：scoped_lock（多个锁）===\n";

    struct Account {
        std::mutex mtx;
        int balance;
        Account(int b) : balance(b) {}
    };

    Account acc1(1000);
    Account acc2(500);

    auto transfer = [](Account& from, Account& to, int amount) {
        // 同时锁定两个账户，自动避免死锁
        std::scoped_lock lock(from.mtx, to.mtx);

        from.balance -= amount;
        to.balance += amount;

        std::cout << "转账 " << amount << " 元成功\n";
    };

    std::thread t1([&]{ transfer(acc1, acc2, 100); });
    std::thread t2([&]{ transfer(acc2, acc1, 50); });

    t1.join();
    t2.join();

    std::cout << "账户1余额: " << acc1.balance << "\n";
    std::cout << "账户2余额: " << acc2.balance << "\n";
    std::cout << "总额: " << (acc1.balance + acc2.balance) << " (应该是 1500)\n";
}

int main() {
    std::cout << "互斥锁演示\n";
    std::cout << "==========\n";

    demo_race_condition();
    demo_manual_lock();
    demo_lock_guard();
    demo_unique_lock();
    demo_lock_granularity();
    demo_scoped_lock();

    std::cout << "\n所有演示完成！\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. 数据竞争：
 *    多个线程同时访问共享数据，至少一个在写 → 必须加锁
 *
 * 2. 三种 RAII 锁：
 *    lock_guard   - 最常用，简单轻量
 *    unique_lock  - 可以手动 lock/unlock，支持条件变量
 *    scoped_lock  - 多个锁，自动避免死锁（C++17）
 *
 * 3. 锁的粒度：
 *    只在访问共享数据时加锁，计算放在锁外
 *
 * 4. 避免死锁：
 *    - 固定加锁顺序
 *    - 用 scoped_lock 同时锁多个
 *
 * 5. 永远不要：
 *    - 手动 lock/unlock（用 RAII）
 *    - 返回被保护数据的引用
 *    - 忘记加锁
 */
