# 内存管理优化

> 理解内存分配、避免内存碎片、提升性能

## 核心概念

**内存分配的代价**：
- `new/delete` 很慢（系统调用）
- 频繁分配 → 内存碎片
- 小对象分配 → 开销大

**优化目标**：
- 减少分配次数
- 复用内存
- 连续内存访问（CPU 缓存友好）

## 内存分配的开销

```cpp
// ❌ 慢：每次循环都 new
for (int i = 0; i < 10000; ++i) {
    int* p = new int(i);  // 10000 次系统调用
    // 使用 p
    delete p;
}

// ✅ 快：一次分配
std::vector<int> vec;
vec.reserve(10000);  // 预分配
for (int i = 0; i < 10000; ++i) {
    vec.push_back(i);  // 无需分配
}
```

**性能对比**：
- 频繁 new/delete：~500ms
- 预分配 vector：~5ms
- **快 100 倍**

## 对象池（Object Pool）

**核心思想**：预先创建对象，使用完归还，不销毁

```cpp
template<typename T>
class ObjectPool {
    std::vector<T*> pool_;       // 可用对象
    std::vector<T*> all_;        // 所有对象
    std::mutex mtx_;

public:
    ObjectPool(size_t size) {
        for (size_t i = 0; i < size; ++i) {
            T* obj = new T();
            pool_.push_back(obj);
            all_.push_back(obj);
        }
    }

    ~ObjectPool() {
        for (T* obj : all_) {
            delete obj;
        }
    }

    // 获取对象
    T* acquire() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pool_.empty()) {
            return nullptr;  // 或扩容
        }
        T* obj = pool_.back();
        pool_.pop_back();
        return obj;
    }

    // 归还对象
    void release(T* obj) {
        std::lock_guard<std::mutex> lock(mtx_);
        pool_.push_back(obj);
    }
};
```

**使用**：
```cpp
ObjectPool<MyObject> pool(100);  // 预创建 100 个

// 获取
MyObject* obj = pool.acquire();
// 使用 obj
obj->do_something();

// 归还（不是 delete）
pool.release(obj);
```

**适用场景**：
- 网络连接池
- 数据库连接池
- 消息对象池

## 内存池（Memory Pool）

**核心思想**：预分配大块内存，手动管理

```cpp
class MemoryPool {
    char* buffer_;       // 大块内存
    size_t size_;
    size_t offset_;      // 当前位置
    std::mutex mtx_;

public:
    MemoryPool(size_t size) : size_(size), offset_(0) {
        buffer_ = new char[size];
    }

    ~MemoryPool() {
        delete[] buffer_;
    }

    // 分配内存
    void* allocate(size_t n) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (offset_ + n > size_) {
            return nullptr;  // 内存不足
        }
        void* ptr = buffer_ + offset_;
        offset_ += n;
        return ptr;
    }

    // 重置（清空）
    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        offset_ = 0;
    }
};
```

**使用**：
```cpp
MemoryPool pool(1024 * 1024);  // 1MB

// 分配
int* p = static_cast<int*>(pool.allocate(sizeof(int)));
*p = 42;

// 使用完后重置
pool.reset();  // 不 delete，直接重用
```

## 小对象分配优化

**问题**：分配很多小对象（如节点）很慢

```cpp
// ❌ 慢：每个节点都 new
struct Node {
    int data;
    Node* next;
};

for (int i = 0; i < 10000; ++i) {
    Node* n = new Node{i, nullptr};  // 慢
}
```

**优化**：分块分配

```cpp
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
    }
};
```

**性能提升**：
- 每次 new：10000 次系统调用
- 分块分配：10 次系统调用
- **快 1000 倍**

## 内存对齐

**CPU 访问对齐内存更快**

```cpp
// ❌ 不对齐（可能慢）
struct BadLayout {
    char c;     // 1 字节
    int i;      // 4 字节
    char c2;    // 1 字节
    double d;   // 8 字节
};
// 实际大小：24 字节（有空洞）

// ✅ 对齐（快）
struct GoodLayout {
    double d;   // 8 字节
    int i;      // 4 字节
    char c;     // 1 字节
    char c2;    // 1 字节
};
// 实际大小：16 字节（紧凑）
```

**规则**：大的放前面，小的放后面

## 避免内存碎片

**问题**：频繁分配/释放 → 内存碎片

```
[占用][空闲][占用][空闲][占用]
      ^^^^        ^^^^
   碎片，浪费，无法使用
```

**解决**：
1. **预分配**：vector::reserve()
2. **对象池**：复用对象
3. **批量分配**：一次分配多个

## RAII 内存管理

```cpp
// 自动管理内存池的 RAII 类
class PooledObject {
    static ObjectPool<PooledObject> pool_;

public:
    // 重载 new
    void* operator new(size_t) {
        return pool_.acquire();
    }

    // 重载 delete
    void operator delete(void* ptr) {
        pool_.release(static_cast<PooledObject*>(ptr));
    }
};

// 使用（和普通对象一样）
PooledObject* obj = new PooledObject();  // 从池获取
delete obj;  // 归还池（不是真正 delete）
```

## 智能指针与内存池

```cpp
// 自定义删除器
template<typename T>
class PoolDeleter {
    ObjectPool<T>* pool_;
public:
    PoolDeleter(ObjectPool<T>* pool) : pool_(pool) {}

    void operator()(T* ptr) {
        pool_->release(ptr);  // 归还而非 delete
    }
};

// 使用
ObjectPool<MyObject> pool(100);
auto obj = std::unique_ptr<MyObject, PoolDeleter<MyObject>>(
    pool.acquire(),
    PoolDeleter<MyObject>(&pool)
);
// 离开作用域自动归还
```

## 性能测量

```cpp
#include <chrono>

// 测量分配性能
auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < 100000; ++i) {
    int* p = new int(i);
    delete p;
}

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "耗时: " << duration.count() << "ms\n";
```

## 常见陷阱

### 陷阱 1：忘记释放对象池

```cpp
// ❌ 忘记归还
MyObject* obj = pool.acquire();
// 使用完忘记 pool.release(obj)
// 池子越来越空

// ✅ RAII 自动归还
{
    auto obj = make_pooled_unique(pool);
    // 使用 obj
}  // 自动归还
```

### 陷阱 2：内存池重置太早

```cpp
// ❌ 危险
void* p = pool.allocate(sizeof(int));
pool.reset();  // 重置了
int* ip = static_cast<int*>(p);
*ip = 42;  // 悬空指针，崩溃

// ✅ 确保不再使用
{
    void* p = pool.allocate(sizeof(int));
    // 使用 p
}
pool.reset();  // 安全
```

### 陷阱 3：对象池容量不足

```cpp
// ❌ 池子太小
ObjectPool<MyObject> pool(10);
for (int i = 0; i < 100; ++i) {
    MyObject* obj = pool.acquire();  // 第 11 次返回 nullptr
}

// ✅ 检查或扩容
MyObject* obj = pool.acquire();
if (!obj) {
    // 扩容或等待
}
```

### 陷阱 4：线程不安全

```cpp
// ❌ 多线程访问池子，未加锁
MyObject* obj = pool.acquire();  // 数据竞争

// ✅ 内部加锁（见上面 ObjectPool 实现）
```

## 实际应用场景

**1. 网络服务器**：
```cpp
// 连接池
ConnectionPool pool(100);
auto conn = pool.acquire();
conn->send(data);
pool.release(conn);
```

**2. 游戏引擎**：
```cpp
// 子弹对象池
ObjectPool<Bullet> bullets(1000);
Bullet* b = bullets.acquire();
b->fire();
bullets.release(b);  // 死亡后回收
```

**3. 数据库**：
```cpp
// 查询结果内存池
MemoryPool pool(10 * 1024 * 1024);  // 10MB
void* result = pool.allocate(query_size);
// 查询完重置
pool.reset();
```

## 核心要点

1. **内存分配很慢**：
   - new/delete 是系统调用
   - 频繁分配 → 性能下降

2. **优化方法**：
   - 预分配（reserve）
   - 对象池（复用对象）
   - 内存池（大块分配）
   - 分块分配（小对象）

3. **对象池 vs 内存池**：
   - 对象池：管理完整对象
   - 内存池：管理原始内存

4. **内存对齐**：
   - 大字段在前，小字段在后
   - CPU 访问更快

5. **避免碎片**：
   - 批量分配
   - 复用内存

6. **性能提升**：
   - 10-1000 倍（取决于场景）

7. **实际应用**：
   - 网络服务器
   - 游戏引擎
   - 数据库系统
