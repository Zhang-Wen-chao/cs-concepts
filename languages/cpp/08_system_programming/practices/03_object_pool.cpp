// 内存管理实践：对象池（Object Pool）
// 编译：g++ -std=c++17 03_object_pool.cpp -o object_pool
// 运行：./object_pool
//
// 目的：演示对象池的实现和性能优势

#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>

// ===== 演示 1：简单对象池 =====

// 模拟一个需要频繁创建的对象
class Message {
public:
    char data[256];
    int size;

    Message() : size(0) {
        // 模拟初始化开销
    }

    void set_data(const char* msg) {
        size = snprintf(data, sizeof(data), "%s", msg);
    }
};

// 简单对象池
template<typename T>
class SimpleObjectPool {
    std::vector<T*> pool_;       // 可用对象
    std::vector<T*> all_;        // 所有对象

public:
    SimpleObjectPool(size_t size) {
        std::cout << "预创建 " << size << " 个对象...\n";
        for (size_t i = 0; i < size; ++i) {
            T* obj = new T();
            pool_.push_back(obj);
            all_.push_back(obj);
        }
    }

    ~SimpleObjectPool() {
        for (T* obj : all_) {
            delete obj;
        }
        std::cout << "对象池销毁，释放 " << all_.size() << " 个对象\n";
    }

    // 获取对象
    T* acquire() {
        if (pool_.empty()) {
            std::cout << "警告：对象池已空\n";
            return nullptr;
        }
        T* obj = pool_.back();
        pool_.pop_back();
        return obj;
    }

    // 归还对象
    void release(T* obj) {
        pool_.push_back(obj);
    }

    size_t available() const { return pool_.size(); }
};

void demo_object_pool() {
    std::cout << "===== 演示 1：对象池基本用法 =====\n";

    SimpleObjectPool<Message> pool(10);
    std::cout << "可用对象数: " << pool.available() << "\n\n";

    // 获取对象
    Message* msg1 = pool.acquire();
    msg1->set_data("Hello");
    std::cout << "获取对象 1，剩余: " << pool.available() << "\n";

    Message* msg2 = pool.acquire();
    msg2->set_data("World");
    std::cout << "获取对象 2，剩余: " << pool.available() << "\n";

    // 归还对象
    pool.release(msg1);
    std::cout << "归还对象 1，剩余: " << pool.available() << "\n";

    pool.release(msg2);
    std::cout << "归还对象 2，剩余: " << pool.available() << "\n\n";
}

// ===== 演示 2：线程安全的对象池 =====

template<typename T>
class ThreadSafeObjectPool {
    std::vector<T*> pool_;
    std::vector<T*> all_;
    std::mutex mtx_;

public:
    ThreadSafeObjectPool(size_t size) {
        for (size_t i = 0; i < size; ++i) {
            T* obj = new T();
            pool_.push_back(obj);
            all_.push_back(obj);
        }
    }

    ~ThreadSafeObjectPool() {
        for (T* obj : all_) {
            delete obj;
        }
    }

    T* acquire() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pool_.empty()) {
            return nullptr;
        }
        T* obj = pool_.back();
        pool_.pop_back();
        return obj;
    }

    void release(T* obj) {
        std::lock_guard<std::mutex> lock(mtx_);
        pool_.push_back(obj);
    }

    size_t available() {
        std::lock_guard<std::mutex> lock(mtx_);
        return pool_.size();
    }
};

void demo_thread_safe_pool() {
    std::cout << "===== 演示 2：线程安全的对象池 =====\n";

    ThreadSafeObjectPool<Message> pool(100);
    std::cout << "创建线程安全对象池，大小: 100\n";
    std::cout << "多线程环境下安全获取/归还对象\n\n";
}

// ===== 演示 3：性能对比 =====

void benchmark_new_delete() {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        Message* msg = new Message();
        msg->set_data("test");
        delete msg;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "new/delete 10000 次: " << duration.count() << " μs\n";
}

void benchmark_object_pool() {
    SimpleObjectPool<Message> pool(100);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        Message* msg = pool.acquire();
        msg->set_data("test");
        pool.release(msg);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "对象池 10000 次: " << duration.count() << " μs\n";
}

void demo_performance() {
    std::cout << "===== 演示 3：性能对比 =====\n";
    benchmark_new_delete();
    benchmark_object_pool();
    std::cout << "\n";
}

// ===== 演示 4：RAII 封装 =====

template<typename T>
class PooledGuard {
    SimpleObjectPool<T>* pool_;
    T* obj_;

public:
    PooledGuard(SimpleObjectPool<T>* pool) : pool_(pool) {
        obj_ = pool_->acquire();
    }

    ~PooledGuard() {
        if (obj_) {
            pool_->release(obj_);
        }
    }

    T* get() { return obj_; }
    T* operator->() { return obj_; }

    PooledGuard(const PooledGuard&) = delete;
    PooledGuard& operator=(const PooledGuard&) = delete;
};

void demo_raii_pool() {
    std::cout << "===== 演示 4：RAII 自动管理 =====\n";

    SimpleObjectPool<Message> pool(10);

    {
        PooledGuard<Message> guard(&pool);
        guard->set_data("RAII");
        std::cout << "使用对象: " << guard->data << "\n";
    }  // 自动归还

    std::cout << "离开作用域后自动归还，剩余: " << pool.available() << "\n\n";
}

int main() {
    demo_object_pool();
    demo_thread_safe_pool();
    demo_performance();
    demo_raii_pool();

    return 0;
}

/*
 * 关键要点：
 *
 * 1. 对象池原理：
 *    - 预创建对象
 *    - acquire() 获取
 *    - release() 归还（不是 delete）
 *
 * 2. 性能优势：
 *    - 避免频繁 new/delete
 *    - 减少系统调用
 *    - 通常快 10-100 倍
 *
 * 3. 线程安全：
 *    - 用 mutex 保护 acquire/release
 *
 * 4. RAII 封装：
 *    - 自动归还对象
 *    - 避免忘记 release
 *
 * 5. 适用场景：
 *    - 网络连接
 *    - 消息对象
 *    - 游戏对象（子弹、特效）
 *
 * 6. 注意事项：
 *    - 池子大小要足够
 *    - 检查 acquire() 返回值
 *    - 确保归还对象
 */
