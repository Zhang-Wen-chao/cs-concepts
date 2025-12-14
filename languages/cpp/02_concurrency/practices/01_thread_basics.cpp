/**
 * Thread Basics Practice
 *
 * 编译：g++ -std=c++17 -pthread 01_thread_basics.cpp -o thread_basics
 * 运行：./thread_basics
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::size_t kTotalNumbers = 1'000'000;
}  // namespace

int main() {
    std::cout << "============================================\n";
    std::cout << "       Thread Basics Practice Demo\n";
    std::cout << "============================================\n";

    // 1. 准备数据
    std::vector<int> data(kTotalNumbers);
    std::iota(data.begin(), data.end(), 1);

    // 2. 决定线程数量
    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) {
        hw_threads = 4;
    }
    const unsigned thread_count = static_cast<unsigned>(
        std::min<std::size_t>(hw_threads, kTotalNumbers));

    std::cout << "hardware concurrency suggested: " << hw_threads << std::endl;
    std::cout << "using threads: " << thread_count << std::endl;

    // 3. 启动线程
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    std::vector<long long> partial_sums(thread_count, 0);
    const std::size_t chunk = (kTotalNumbers + thread_count - 1) / thread_count;
    std::mutex log_mutex;

    auto start = std::chrono::steady_clock::now();

    for (unsigned i = 0; i < thread_count; ++i) {
        const std::size_t begin = i * chunk;
        const std::size_t end = std::min(begin + chunk, kTotalNumbers);
        if (begin >= end) {
            break;
        }

        workers.emplace_back([&, i, begin, end] {
            long long local = 0;
            for (std::size_t idx = begin; idx < end; ++idx) {
                local += data[idx];
            }
            partial_sums[i] = local;

            const auto tid = std::this_thread::get_id();
            std::lock_guard<std::mutex> guard(log_mutex);
            std::cout << "[thread " << std::setw(2) << i << "] id=" << tid
                      << ", range=[" << begin << ", " << end << ")"
                      << ", partial=" << local << std::endl;
        });
    }

    // 4. 等待线程完成
    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto stop = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    // 5. 汇总结果
    long long total = 0;
    for (auto part : partial_sums) {
        total += part;
    }

    const long long expected = static_cast<long long>(kTotalNumbers) *
                               (static_cast<long long>(kTotalNumbers) + 1) / 2;

    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Total sum = " << total << std::endl;
    std::cout << "Expected  = " << expected << std::endl;
    std::cout << "Match?    = " << std::boolalpha << (total == expected) << std::endl;
    std::cout << "Elapsed   = " << elapsed_ms << " ms" << std::endl;
    std::cout << "============================================\n";

    return (total == expected) ? 0 : 1;
}
