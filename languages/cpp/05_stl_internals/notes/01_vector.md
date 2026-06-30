# vector 源码探秘

## 1. 内存布局

`std::vector<T>` 核心是三个指针（在 libstdc++ 实现中）：

```cpp
template <typename T>
class vector {
    T* start_;     // begin()
    T* finish_;    // end() — 真正存储的结束
    T* end_of_storage_;  // capacity() — 分配内存的结束
};
```

```
[ T | T | T | . | . ]    ← 已分配内存
  ↑        ↑       ↑
  start_   finish_  end_of_storage_
  (begin)  (end)    (capacity)
```

## 2. 动态扩容

```cpp
void push_back(const T& value) {
    if (finish_ != end_of_storage_) {
        // 还有空间：原地构造
        new (finish_) T(value);   // placement new
        ++finish_;
    } else {
        // 空间满了：重新分配 2 倍大小
        size_t new_cap = capacity() == 0 ? 1 : 2 * capacity();
        T* new_data = allocate(new_cap);
        
        // 移动/拷贝旧元素到新内存
        for (size_t i = 0; i < size(); ++i)
            new (new_data + i) T(std::move_if_noexcept(start_[i]));
        
        // 析构旧元素，释放旧内存
        for (size_t i = 0; i < size(); ++i)
            start_[i].~T();
        deallocate(start_);
        
        // 更新指针
        start_ = new_data;
        finish_ = start_ + old_size;
        end_of_storage_ = start_ + new_cap;
    }
}
```

**增长因子**：通常为 2（Visual Studio）或 1.5（libstdc++）。2 倍的问题：无法复用之前的内存块（斐波那契数列 1.618 更好）。

## 3. 迭代器失效

```cpp
std::vector<int> v = {1,2,3,4,5};
int& ref = v[2];
auto it  = v.begin() + 2;

v.push_back(6);         // 如果触发扩容 → ref 和 it 都失效
v.insert(v.begin(), 0); // 始终失效（所有元素移动）
v.erase(v.begin());     // 被删元素及之后的所有迭代器失效
v.erase(v.begin(), v.begin()+2); // 同上
```

## 4. emplace_back 优于 push_back

```cpp
// push_back: 先构造临时对象再移动/拷贝
v.push_back(std::pair<int, std::string>(1, "hello"));

// emplace_back: 直接就地构造，零拷贝
v.emplace_back(1, "hello");
```

`emplace` 系列把参数完美转发给构造函数，避免临时对象开销。

## 关键点总结

- vector 是 **动态数组**，内存连续 O(1) 随机访问
- 扩容是 O(n)，但**均摊 O(1)**（扩容次数随大小指数减少）
- 扩容后所有**迭代器、引用、指针**失效
- `emplace_back` 优于 `push_back`（少一次移动构造）
- `reserve(n)` 预分配可避免多次扩容
