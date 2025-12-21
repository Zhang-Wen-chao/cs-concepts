// 内存管理实践：内存池（Memory Pool）
// 编译：g++ -std=c++17 03_memory_pool.cpp -o memory_pool
// 运行：./memory_pool
//
// 目的：演示内存池的实现和使用

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

// ===== 演示 1：简单内存池 =====

class SimpleMemoryPool {
    char* buffer_;       // 大块内存
    size_t size_;        // 总大小
    size_t offset_;      // 当前位置

public:
    SimpleMemoryPool(size_t size) : size_(size), offset_(0) {
        buffer_ = new char[size];
        std::cout << "内存池创建: " << size << " 字节\n";
    }

    ~SimpleMemoryPool() {
        delete[] buffer_;
        std::cout << "内存池销毁\n";
    }

    // 分配内存
    void* allocate(size_t n) {
        if (offset_ + n > size_) {
            std::cout << "内存不足！请求 " << n << " 字节，剩余 " << (size_ - offset_) << " 字节\n";
            return nullptr;
        }
        void* ptr = buffer_ + offset_;
        offset_ += n;
        return ptr;
    }

    // 重置（清空）
    void reset() {
        offset_ = 0;
        std::cout << "内存池已重置\n";
    }

    size_t used() const { return offset_; }
    size_t available() const { return size_ - offset_; }
};

void demo_memory_pool() {
    std::cout << "===== 演示 1：简单内存池 =====\n";

    SimpleMemoryPool pool(1024);  // 1KB

    // 分配一些内存
    int* p1 = static_cast<int*>(pool.allocate(sizeof(int)));
    *p1 = 42;
    std::cout << "分配 int，值: " << *p1 << "，已用: " << pool.used() << " 字节\n";

    double* p2 = static_cast<double*>(pool.allocate(sizeof(double)));
    *p2 = 3.14;
    std::cout << "分配 double，值: " << *p2 << "，已用: " << pool.used() << " 字节\n";

    char* str = static_cast<char*>(pool.allocate(20));
    strcpy(str, "Hello");
    std::cout << "分配字符串: " << str << "，已用: " << pool.used() << " 字节\n";

    std::cout << "剩余: " << pool.available() << " 字节\n\n";

    // 重置
    pool.reset();
    std::cout << "重置后，已用: " << pool.used() << " 字节\n\n";
}

// ===== 演示 2：对齐的内存池 =====

class AlignedMemoryPool {
    char* buffer_;
    size_t size_;
    size_t offset_;

public:
    AlignedMemoryPool(size_t size) : size_(size), offset_(0) {
        buffer_ = new char[size];
    }

    ~AlignedMemoryPool() {
        delete[] buffer_;
    }

    // 对齐分配
    void* allocate(size_t n, size_t alignment = 8) {
        // 计算对齐后的位置
        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + n > size_) {
            return nullptr;
        }

        void* ptr = buffer_ + aligned_offset;
        offset_ = aligned_offset + n;
        return ptr;
    }

    void reset() { offset_ = 0; }
};

void demo_aligned_pool() {
    std::cout << "===== 演示 2：对齐的内存池 =====\n";

    AlignedMemoryPool pool(1024);

    // 分配对齐内存
    void* p1 = pool.allocate(1, 8);   // 1 字节，8 字节对齐
    void* p2 = pool.allocate(4, 8);   // 4 字节，8 字节对齐

    std::cout << "p1 地址: " << p1 << "\n";
    std::cout << "p2 地址: " << p2 << "\n";
    std::cout << "地址差: " << (static_cast<char*>(p2) - static_cast<char*>(p1)) << " 字节\n\n";
}

// ===== 演示 3：性能对比 =====

void benchmark_malloc() {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        void* p = malloc(32);
        free(p);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "malloc/free 10000 次: " << duration.count() << " μs\n";
}

void benchmark_memory_pool() {
    SimpleMemoryPool pool(1024 * 1024);  // 1MB

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        void* p = pool.allocate(32);
        if (i % 100 == 0) {
            pool.reset();  // 每 100 次重置
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "内存池 10000 次: " << duration.count() << " μs\n";
}

void demo_performance() {
    std::cout << "===== 演示 3：性能对比 =====\n";
    benchmark_malloc();
    benchmark_memory_pool();
    std::cout << "\n";
}

// ===== 演示 4：分块分配（用于小对象）=====

struct Node {
    int data;
    Node* next;
};

class NodeAllocator {
    static constexpr size_t CHUNK_SIZE = 1024;
    std::vector<Node*> chunks_;
    size_t current_chunk_ = 0;
    size_t current_index_ = 0;

public:
    NodeAllocator() {
        allocate_chunk();
    }

    ~NodeAllocator() {
        for (Node* chunk : chunks_) {
            delete[] chunk;
        }
        std::cout << "释放 " << chunks_.size() << " 个块\n";
    }

    Node* allocate() {
        if (current_index_ >= CHUNK_SIZE) {
            allocate_chunk();
        }
        return &chunks_[current_chunk_][current_index_++];
    }

private:
    void allocate_chunk() {
        chunks_.push_back(new Node[CHUNK_SIZE]);
        current_chunk_ = chunks_.size() - 1;
        current_index_ = 0;
        std::cout << "分配新块，共 " << chunks_.size() << " 个\n";
    }
};

void demo_node_allocator() {
    std::cout << "===== 演示 4：分块分配 =====\n";

    NodeAllocator allocator;

    // 分配很多节点
    Node* head = nullptr;
    for (int i = 0; i < 2000; ++i) {
        Node* node = allocator.allocate();
        node->data = i;
        node->next = head;
        head = node;
    }

    std::cout << "分配了 2000 个节点\n";
    std::cout << "前 3 个节点: " << head->data << " -> "
              << head->next->data << " -> "
              << head->next->next->data << "\n\n";
}

// ===== 演示 5：内存池用于临时数据 =====

void demo_temporary_data() {
    std::cout << "===== 演示 5：临时数据分配 =====\n";

    SimpleMemoryPool pool(4096);  // 4KB

    // 处理多个请求
    for (int req = 0; req < 3; ++req) {
        std::cout << "处理请求 " << req << ":\n";

        // 分配临时缓冲区
        char* buffer = static_cast<char*>(pool.allocate(512));
        snprintf(buffer, 512, "请求 %d 的数据", req);
        std::cout << "  " << buffer << "\n";

        // 请求完成，重置内存池
        pool.reset();
        std::cout << "  内存池已重置\n";
    }

    std::cout << "\n";
}

int main() {
    demo_memory_pool();
    demo_aligned_pool();
    demo_performance();
    demo_node_allocator();
    demo_temporary_data();

    return 0;
}

/*
 * 关键要点：
 *
 * 1. 内存池原理：
 *    - 预分配大块内存
 *    - allocate() 返回指针，offset 递增
 *    - reset() 重置 offset，复用内存
 *
 * 2. 对齐：
 *    - CPU 访问对齐地址更快
 *    - 通常对齐到 8 字节
 *
 * 3. 性能优势：
 *    - 避免系统调用
 *    - 通常快 10-100 倍
 *
 * 4. 分块分配：
 *    - 适合大量小对象
 *    - 每次分配 1024 个
 *    - 避免频繁分配
 *
 * 5. 适用场景：
 *    - 临时数据（请求/响应）
 *    - 频繁分配的小对象
 *    - 已知生命周期的数据
 *
 * 6. 限制：
 *    - 无法单独释放
 *    - 只能整体 reset
 *    - 需要预估大小
 *
 * 7. 对象池 vs 内存池：
 *    - 对象池：管理完整对象，可单独归还
 *    - 内存池：管理原始内存，整体重置
 */
