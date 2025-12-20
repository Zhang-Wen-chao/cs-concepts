// 线程池实践：复用线程高效执行任务
// 编译：g++ -std=c++17 -pthread 06_thread_pool.cpp -o thread_pool
// 运行：./thread_pool
//
// 目的：演示线程池的实现和使用

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <chrono>

// 简单线程池实现
class ThreadPool {
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;

public:
    ThreadPool(size_t threads) : stop_(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();

        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>
    {
        using return_type = typename std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> res = task->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("线程池已停止");
            }
            tasks_.emplace([task]() { (*task)(); });
        }

        condition_.notify_one();
        return res;
    }
};

// 演示 1：基本使用
void demo_basic() {
    std::cout << "\n=== 演示 1：基本使用 ===\n";

    ThreadPool pool(4);

    for (int i = 0; i < 10; ++i) {
        pool.submit([i] {
            std::cout << "任务 " << i << " 在线程 "
                      << std::this_thread::get_id() << " 执行\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        });
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "所有任务已提交\n";
}

// 演示 2：获取返回值
void demo_return_value() {
    std::cout << "\n=== 演示 2：获取返回值 ===\n";

    ThreadPool pool(4);

    std::vector<std::future<int>> results;

    for (int i = 0; i < 10; ++i) {
        results.push_back(pool.submit([i] {
            return i * i;
        }));
    }

    std::cout << "计算结果: ";
    for (auto& result : results) {
        std::cout << result.get() << " ";
    }
    std::cout << "\n";
}

// 演示 3：性能对比（线程池 vs 直接创建线程）
void demo_performance() {
    std::cout << "\n=== 演示 3：性能对比 ===\n";

    const int TASKS = 1000;

    // 直接创建线程
    {
        auto start = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        for (int i = 0; i < TASKS; ++i) {
            threads.emplace_back([i] {
                // 模拟任务
                volatile int sum = 0;
                for (int j = 0; j < 1000; ++j) sum += j;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "直接创建线程: " << elapsed.count() << " ms\n";
    }

    // 线程池
    {
        auto start = std::chrono::steady_clock::now();

        ThreadPool pool(std::thread::hardware_concurrency());
        std::vector<std::future<void>> futures;

        for (int i = 0; i < TASKS; ++i) {
            futures.push_back(pool.submit([i] {
                // 模拟任务
                volatile int sum = 0;
                for (int j = 0; j < 1000; ++j) sum += j;
            }));
        }

        for (auto& fut : futures) {
            fut.get();
        }

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "线程池:       " << elapsed.count() << " ms\n";
    }

    std::cout << "结论：线程池更快（避免频繁创建/销毁线程）\n";
}

// 演示 4：并行计算斐波那契数列
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

void demo_parallel_fibonacci() {
    std::cout << "\n=== 演示 4：并行计算斐波那契数列 ===\n";

    ThreadPool pool(std::thread::hardware_concurrency());

    std::vector<std::future<int>> futures;

    for (int i = 30; i <= 35; ++i) {
        futures.push_back(pool.submit(fibonacci, i));
    }

    std::cout << "斐波那契数列 (30-35):\n";
    for (size_t i = 0; i < futures.size(); ++i) {
        std::cout << "F(" << (30 + i) << ") = " << futures[i].get() << "\n";
    }
}

// 演示 5：任务依赖
void demo_task_dependency() {
    std::cout << "\n=== 演示 5：任务依赖 ===\n";

    ThreadPool pool(4);

    // 阶段 1：计算
    auto fut1 = pool.submit([] {
        std::cout << "阶段 1：计算中...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return 10;
    });

    auto fut2 = pool.submit([] {
        std::cout << "阶段 1：计算中...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return 20;
    });

    // 阶段 2：依赖阶段 1 的结果
    int result1 = fut1.get();
    int result2 = fut2.get();

    auto fut3 = pool.submit([result1, result2] {
        std::cout << "阶段 2：合并结果...\n";
        return result1 + result2;
    });

    std::cout << "最终结果: " << fut3.get() << "\n";
}

// 演示 6：批量处理
void demo_batch_processing() {
    std::cout << "\n=== 演示 6：批量处理 ===\n";

    ThreadPool pool(4);

    // 模拟处理 100 个数据项
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = i;
    }

    // 分批提交
    const int BATCH_SIZE = 10;
    std::vector<std::future<int>> futures;

    for (size_t i = 0; i < data.size(); i += BATCH_SIZE) {
        size_t end = std::min(i + BATCH_SIZE, data.size());

        futures.push_back(pool.submit([&data, i, end] {
            int sum = 0;
            for (size_t j = i; j < end; ++j) {
                sum += data[j];
            }
            return sum;
        }));
    }

    // 汇总结果
    int total = 0;
    for (auto& fut : futures) {
        total += fut.get();
    }

    std::cout << "总和: " << total << " (期望 4950)\n";
}

int main() {
    std::cout << "线程池演示\n";
    std::cout << "==========\n";
    std::cout << "CPU 核心数: " << std::thread::hardware_concurrency() << "\n";

    demo_basic();
    demo_return_value();
    demo_performance();
    demo_parallel_fibonacci();
    demo_task_dependency();
    demo_batch_processing();

    std::cout << "\n所有演示完成！\n";
    return 0;
}

/*
 * 关键要点：
 *
 * 1. 线程池结构：
 *    - 工作线程（预先创建）
 *    - 任务队列
 *    - 互斥锁（保护队列）
 *    - 条件变量（通知新任务）
 *
 * 2. 提交任务：
 *    auto fut = pool.submit(task);
 *    int result = fut.get();
 *
 * 3. 线程池大小：
 *    - CPU 密集型：CPU 核心数
 *    - I/O 密集型：CPU 核心数 × 2
 *
 * 4. 优势：
 *    - 复用线程，避免创建/销毁开销
 *    - 限制并发数量
 *    - 易于管理
 *
 * 5. 注意：
 *    - 避免任务互相等待（死锁）
 *    - 限制任务队列大小
 *    - 确保任务完成后再析构
 *
 * 6. 使用 packaged_task：
 *    - 包装任务
 *    - 通过 future 获取返回值
 *
 * 7. 实际应用：
 *    - Web 服务器
 *    - 图像处理
 *    - 数据处理
 */
