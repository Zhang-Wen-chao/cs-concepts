// 条件变量实践：生产者-消费者模型
// 编译：g++ -std=c++17 -pthread 03_condition_variable.cpp -o condition_variable
// 运行：./condition_variable
//
// 目的：演示条件变量的等待/通知机制

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

// 演示 1：基本的等待/通知
void demo_basic() {
    std::cout << "\n=== 演示 1：基本等待/通知 ===\n";

    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;

    // 等待线程
    std::thread waiter([&]{
        std::cout << "等待线程：开始等待...\n";

        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return ready; });  // 阻塞，直到 ready 为 true

        std::cout << "等待线程：收到通知，继续执行\n";
    });

    // 通知线程
    std::thread notifier([&]{
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::cout << "通知线程：设置 ready = true\n";
        {
            std::lock_guard<std::mutex> lock(mtx);
            ready = true;
        }

        std::cout << "通知线程：发送通知\n";
        cv.notify_one();
    });

    waiter.join();
    notifier.join();
}

// 演示 2：生产者-消费者（单生产单消费）
void demo_producer_consumer() {
    std::cout << "\n=== 演示 2：生产者-消费者 ===\n";

    std::queue<int> buffer;
    std::mutex mtx;
    std::condition_variable cv;
    const int MAX_SIZE = 5;
    const int TOTAL = 10;

    // 生产者
    std::thread producer([&]{
        for (int i = 0; i < TOTAL; ++i) {
            std::unique_lock<std::mutex> lock(mtx);

            // 等待缓冲区不满
            cv.wait(lock, [&]{ return buffer.size() < MAX_SIZE; });

            buffer.push(i);
            std::cout << "生产: " << i << " (缓冲区: " << buffer.size() << ")\n";

            lock.unlock();
            cv.notify_one();  // 通知消费者

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // 消费者
    std::thread consumer([&]{
        for (int i = 0; i < TOTAL; ++i) {
            std::unique_lock<std::mutex> lock(mtx);

            // 等待缓冲区不空
            cv.wait(lock, [&]{ return !buffer.empty(); });

            int value = buffer.front();
            buffer.pop();
            std::cout << "消费: " << value << " (缓冲区: " << buffer.size() << ")\n";

            lock.unlock();
            cv.notify_one();  // 通知生产者

            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
    });

    producer.join();
    consumer.join();
}

// 演示 3：多生产者-多消费者
void demo_multi_producers_consumers() {
    std::cout << "\n=== 演示 3：多生产者-多消费者 ===\n";

    std::queue<int> buffer;
    std::mutex mtx;
    std::condition_variable cv;
    const int MAX_SIZE = 3;
    bool done = false;

    // 生产者函数
    auto producer = [&](int id, int count) {
        for (int i = 0; i < count; ++i) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]{ return buffer.size() < MAX_SIZE; });

            int value = id * 100 + i;
            buffer.push(value);
            std::cout << "生产者 " << id << " 生产: " << value << "\n";

            lock.unlock();
            cv.notify_all();  // 通知所有消费者

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    };

    // 消费者函数
    auto consumer = [&](int id) {
        while (true) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]{ return !buffer.empty() || done; });

            if (buffer.empty() && done) break;

            int value = buffer.front();
            buffer.pop();
            std::cout << "消费者 " << id << " 消费: " << value << "\n";

            lock.unlock();
            cv.notify_all();  // 通知所有生产者

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };

    // 创建 2 个生产者，3 个消费者
    std::thread p1(producer, 1, 5);
    std::thread p2(producer, 2, 5);
    std::thread c1(consumer, 1);
    std::thread c2(consumer, 2);
    std::thread c3(consumer, 3);

    p1.join();
    p2.join();

    // 生产完成，通知消费者结束
    {
        std::lock_guard<std::mutex> lock(mtx);
        done = true;
    }
    cv.notify_all();

    c1.join();
    c2.join();
    c3.join();
}

// 演示 4：wait_for（带超时）
void demo_wait_for() {
    std::cout << "\n=== 演示 4：wait_for（超时）===\n";

    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;

    std::thread waiter([&]{
        std::unique_lock<std::mutex> lock(mtx);

        std::cout << "等待 2 秒...\n";
        bool result = cv.wait_for(lock, std::chrono::seconds(2), [&]{ return ready; });

        if (result) {
            std::cout << "条件满足\n";
        } else {
            std::cout << "超时，条件未满足\n";
        }
    });

    // 不发送通知，让它超时
    waiter.join();
}

// 演示 5：对比忙等 vs 条件变量
void demo_busy_wait_vs_cv() {
    std::cout << "\n=== 演示 5：忙等 vs 条件变量 ===\n";

    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;

    // ❌ 忙等（浪费 CPU）
    std::cout << "忙等方式（不推荐）：\n";
    std::thread busy_waiter([&]{
        auto start = std::chrono::steady_clock::now();
        while (true) {
            std::lock_guard<std::mutex> lock(mtx);
            if (ready) break;
            // CPU 不断循环检查
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "忙等耗时: " << elapsed.count() << " ms\n";
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    busy_waiter.join();

    // ✅ 条件变量（高效）
    ready = false;
    std::cout << "\n条件变量方式（推荐）：\n";
    std::thread cv_waiter([&]{
        auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return ready; });  // 阻塞，不占 CPU
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "条件变量耗时: " << elapsed.count() << " ms\n";
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one();
    cv_waiter.join();

    std::cout << "条件变量不占用 CPU，效率更高\n";
}

int main() {
    std::cout << "条件变量演示\n";
    std::cout << "============\n";

    demo_basic();
    demo_producer_consumer();
    demo_multi_producers_consumers();
    demo_wait_for();
    demo_busy_wait_vs_cv();

    std::cout << "\n所有演示完成！\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. 条件变量：线程间等待/通知机制
 *    std::condition_variable cv;
 *
 * 2. 等待：
 *    std::unique_lock<std::mutex> lock(mtx);
 *    cv.wait(lock, []{ return condition; });  // 必须用谓词
 *
 * 3. 通知：
 *    cv.notify_one();   // 唤醒一个线程
 *    cv.notify_all();   // 唤醒所有线程
 *
 * 4. 必须用 unique_lock：
 *    wait 需要临时解锁，lock_guard 不支持
 *
 * 5. 总是用谓词：
 *    避免虚假唤醒
 *
 * 6. 经典模式：生产者-消费者
 *    用队列 + mutex + condition_variable
 *
 * 7. 性能：
 *    条件变量 >> 忙等（不占 CPU）
 */
