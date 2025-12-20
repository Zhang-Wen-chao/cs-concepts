// 异步编程实践：async/future/promise
// 编译：g++ -std=c++17 -pthread 05_async_future.cpp -o async_demo
// 运行：./async_demo
//
// 目的：演示异步编程的各种方式

#include <iostream>
#include <future>
#include <thread>
#include <chrono>
#include <vector>

// 演示 1：std::async 基本用法
void demo_async_basic() {
    std::cout << "\n=== 演示 1：std::async 基本用法 ===\n";

    // 启动异步任务
    auto fut = std::async(std::launch::async, []{
        std::cout << "异步任务开始\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "异步任务完成\n";
        return 42;
    });

    std::cout << "主线程继续执行...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 获取结果（阻塞）
    std::cout << "等待结果...\n";
    int result = fut.get();
    std::cout << "结果: " << result << "\n";
}

// 演示 2：async vs deferred
void demo_launch_policy() {
    std::cout << "\n=== 演示 2：启动策略 ===\n";

    // async：立即创建线程
    auto fut1 = std::async(std::launch::async, []{
        std::cout << "async: 立即在新线程执行\n";
        return 1;
    });

    // deferred：延迟执行（调用 get 时才执行）
    auto fut2 = std::async(std::launch::deferred, []{
        std::cout << "deferred: 调用 get 时执行\n";
        return 2;
    });

    std::cout << "主线程做其他事...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "\n调用 fut2.get()...\n";
    fut2.get();  // 现在才执行 deferred 任务

    fut1.get();
}

// 演示 3：future 的等待方法
void demo_future_wait() {
    std::cout << "\n=== 演示 3：future 等待方法 ===\n";

    auto fut = std::async(std::launch::async, []{
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return 42;
    });

    // wait_for：等待一段时间
    std::cout << "等待 1 秒...\n";
    auto status = fut.wait_for(std::chrono::seconds(1));

    if (status == std::future_status::timeout) {
        std::cout << "超时，任务还在执行\n";
    }

    // 再等待
    std::cout << "继续等待...\n";
    fut.wait();  // 阻塞，直到完成
    std::cout << "任务完成，结果: " << fut.get() << "\n";
}

// 演示 4：std::promise 手动设置结果
void demo_promise() {
    std::cout << "\n=== 演示 4：promise 手动设置结果 ===\n";

    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    // 生产者线程
    std::thread producer([&prom]{
        std::cout << "生产者：计算中...\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
        prom.set_value(100);  // 设置结果
        std::cout << "生产者：结果已设置\n";
    });

    // 消费者线程
    std::cout << "消费者：等待结果...\n";
    int result = fut.get();  // 阻塞，直到 promise 设置值
    std::cout << "消费者：收到结果 " << result << "\n";

    producer.join();
}

// 演示 5：promise 设置异常
void demo_promise_exception() {
    std::cout << "\n=== 演示 5：promise 设置异常 ===\n";

    std::promise<int> prom;
    std::future<int> fut = prom.get_future();

    std::thread t([&prom]{
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            throw std::runtime_error("计算出错");
        } catch (...) {
            prom.set_exception(std::current_exception());
        }
    });

    try {
        std::cout << "等待结果...\n";
        fut.get();  // 抛出异常
    } catch (const std::exception& e) {
        std::cout << "捕获异常: " << e.what() << "\n";
    }

    t.join();
}

// 演示 6：std::packaged_task
void demo_packaged_task() {
    std::cout << "\n=== 演示 6：packaged_task ===\n";

    // 包装函数
    std::packaged_task<int(int, int)> task([](int a, int b) {
        std::cout << "packaged_task 执行中...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return a + b;
    });

    // 获取 future
    std::future<int> fut = task.get_future();

    // 在线程中执行
    std::thread t(std::move(task), 10, 20);

    std::cout << "主线程等待结果...\n";
    int result = fut.get();
    std::cout << "结果: " << result << "\n";

    t.join();
}

// 演示 7：shared_future（多个消费者）
void demo_shared_future() {
    std::cout << "\n=== 演示 7：shared_future ===\n";

    std::future<int> fut = std::async(std::launch::async, []{
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 42;
    });

    // 转换为 shared_future
    std::shared_future<int> sf = fut.share();

    // 多个线程都可以获取结果
    std::thread t1([sf]{
        std::cout << "线程 1 获取: " << sf.get() << "\n";
    });

    std::thread t2([sf]{
        std::cout << "线程 2 获取: " << sf.get() << "\n";
    });

    std::thread t3([sf]{
        std::cout << "线程 3 获取: " << sf.get() << "\n";
    });

    t1.join();
    t2.join();
    t3.join();
}

// 演示 8：并行计算
void demo_parallel_computation() {
    std::cout << "\n=== 演示 8：并行计算 ===\n";

    // 模拟计算任务
    auto compute = [](int id) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return id * id;
    };

    // 启动多个异步任务
    std::vector<std::future<int>> futures;
    for (int i = 1; i <= 5; ++i) {
        futures.push_back(std::async(std::launch::async, compute, i));
    }

    std::cout << "所有任务已启动，主线程继续执行\n";

    // 收集结果
    std::cout << "收集结果:\n";
    for (size_t i = 0; i < futures.size(); ++i) {
        int result = futures[i].get();
        std::cout << "任务 " << (i + 1) << " 结果: " << result << "\n";
    }
}

// 演示 9：async vs thread 对比
void demo_async_vs_thread() {
    std::cout << "\n=== 演示 9：async vs thread 对比 ===\n";

    // 用 thread（复杂）
    std::cout << "使用 thread:\n";
    int result1;
    std::thread t([&result1]{
        result1 = 42;
    });
    t.join();
    std::cout << "结果: " << result1 << "\n";

    // 用 async（简洁）
    std::cout << "\n使用 async:\n";
    auto fut = std::async([]{ return 42; });
    int result2 = fut.get();
    std::cout << "结果: " << result2 << "\n";

    std::cout << "\nasync 更简洁！\n";
}

int main() {
    std::cout << "异步编程演示\n";
    std::cout << "============\n";

    demo_async_basic();
    demo_launch_policy();
    demo_future_wait();
    demo_promise();
    demo_promise_exception();
    demo_packaged_task();
    demo_shared_future();
    demo_parallel_computation();
    demo_async_vs_thread();

    std::cout << "\n所有演示完成！\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. std::async - 启动异步任务（最简单）
 *    auto fut = std::async([]{ return 42; });
 *    int result = fut.get();
 *
 * 2. 启动策略：
 *    std::launch::async    - 立即创建线程
 *    std::launch::deferred - 延迟执行（调用 get 时）
 *
 * 3. std::future - 获取结果
 *    fut.get()       - 阻塞获取结果（只能调用一次）
 *    fut.wait()      - 等待完成（不获取结果）
 *    fut.wait_for()  - 等待一段时间
 *
 * 4. std::promise - 手动设置结果
 *    prom.set_value(42)            - 设置值
 *    prom.set_exception(...)       - 设置异常
 *
 * 5. std::packaged_task - 包装函数
 *    用于线程池等场景
 *
 * 6. std::shared_future - 多个消费者
 *    可以多次调用 get()
 *
 * 7. 选择：
 *    - 简单异步 → async
 *    - 精确控制 → thread
 *    - 手动控制结果 → promise
 *
 * 8. 注意：
 *    - future.get() 只能调用一次
 *    - future 析构会阻塞
 *    - promise 必须设置值
 */
