# 智能指针

> 告别裸指针，自动内存管理

## 裸指针的三大问题

1. **所有权不清晰**：`void foo(int* p)` 谁负责释放？
2. **容易忘记释放**：异常或提前 return 导致内存泄漏
3. **悬空指针**：`delete p; *p = 20;` 访问已释放内存

## 三种智能指针

### unique_ptr（90% 的情况用它）

**独占所有权，不能拷贝，只能移动**

```cpp
// 创建方式1：make_unique（推荐，异常安全）
auto p1 = std::make_unique<int>(42);

// 创建方式2：直接构造（也可以）
std::unique_ptr<int> p2(new int(42));

// 注意：make_unique 不是必须的，但更推荐
//      make_unique = 更安全 + 更简洁

// 不能拷贝
// auto p2 = p1;  // ❌ 编译错误

// 可以移动
auto p2 = std::move(p1);  // p1 变空，p2 拥有资源

// 访问
*p2 = 100;
int* raw = p2.get();  // 获取原始指针

// 数组
auto arr = std::make_unique<int[]>(100);
arr[0] = 42;
```

### shared_ptr（需要共享时用）

**共享所有权，引用计数，最后一个销毁时释放**

```cpp
// 创建方式1：make_shared（推荐，性能更好）
auto p1 = std::make_shared<int>(42);

// 创建方式2：直接构造（不推荐，两次内存分配）
std::shared_ptr<int> p2(new int(42));

// make_shared 的优势：一次内存分配（对象+控制块）
// 直接构造：两次内存分配（对象 + 控制块分开）

// 拷贝（引用计数 +1）
auto p2 = p1;  // 引用计数 = 2

// 查询引用计数
std::cout << p1.use_count();  // 2

// 重置（引用计数 -1）
p1.reset();  // 引用计数 = 1
```

### weak_ptr（打破循环引用）

**不拥有对象，不增加引用计数**

```cpp
// 循环引用问题
struct Node {
    std::shared_ptr<Node> next;  // 强引用
    std::weak_ptr<Node> prev;    // 弱引用，打破循环
};

// 使用 weak_ptr
auto sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;  // 不增加引用计数

// 访问前先转换为 shared_ptr
if (auto temp = wp.lock()) {
    std::cout << *temp;  // 安全访问
} else {
    std::cout << "对象已释放";
}
```

## 选择指南

```
需要动态内存？
    ↓
独占所有权？ → unique_ptr（默认选择）
    ↓
需要共享？ → shared_ptr
    ↓
有循环引用？ → weak_ptr 打破循环
```

## 常见陷阱

```cpp
// ❌ 不要用同一个裸指针初始化多个智能指针
int* raw = new int(42);
std::unique_ptr<int> p1(raw);
std::unique_ptr<int> p2(raw);  // 重复释放

// ❌ 不要从智能指针获取裸指针后再创建智能指针
auto p1 = std::make_unique<int>(42);
std::unique_ptr<int> p2(p1.get());  // 重复释放

// ❌ 函数参数不要传值（引用计数开销）
void foo(std::shared_ptr<T> p);  // 不好
void foo(const std::shared_ptr<T>& p);  // 好
void foo(T* p);  // 更好（不需要所有权时）
```

## 最佳实践

```cpp
// 1. 默认用 unique_ptr
auto p = std::make_unique<T>();

// 2. 需要共享才用 shared_ptr
auto sp = std::make_shared<T>();

// 3. 用 make_unique/make_shared（不要手动 new）
// ✅ auto p = std::make_unique<T>();
// ❌ std::unique_ptr<T> p(new T());

// 4. 函数参数按需传递
void use_only(T& obj);                    // 只使用
void take_ownership(std::unique_ptr<T> p); // 转移所有权
void share(std::shared_ptr<T> p);          // 共享所有权
```

## 性能对比

| 类型 | 大小 | 开销 | 使用场景 |
|-----|------|------|---------|
| unique_ptr | sizeof(T*) | 零开销 | 90% 的情况 |
| shared_ptr | 2*sizeof(T*) | 引用计数 | 需要共享 |
| weak_ptr | 2*sizeof(T*) | 小 | 打破循环 |
