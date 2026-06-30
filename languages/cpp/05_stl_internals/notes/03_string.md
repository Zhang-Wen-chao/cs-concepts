# string 源码探秘

## 1. std::string 的本质

`std::string` 是 `std::basic_string<char>` 的别名：

```cpp
using string = basic_string<char>;
```

`basic_string` 是一个**管理字符缓冲区的 RAII 容器**，和 vector 非常类似。

## 2. SSO（小字符串优化）

为了减少堆分配，大多数实现使用 **SSO（Small String Optimization）**：

```cpp
template <typename CharT>
class basic_string {
    // libstdc++ 实现思路
    union {
        struct {        // 堆分配模式
            CharT* data_;
            size_t size_;
            size_t capacity_;
        };
        CharT local_[16];  // SSO 缓冲区（栈上）
    };
    // 用一个字节标记当前是 SSO 还是堆模式
    bool use_sso_;
};
```

**SSO 效果**：
```
string s = "hi";      // → 存在栈上 local_[16]，零堆分配
string s = "这是一段非常长的字符串，超过15个字符了";  // → 触发堆分配
```

不同实现的 SSO 缓冲区大小：
- libstdc++：15 字节（含末尾 \0）
- libc++：22 字节
- MSVC：16 字节

## 3. COW（写时复制）— 已废弃

C++11 之前，某些实现（如 GCC）用 **COW（Copy-On-Write）**：多个 string 共享同一缓冲区，写的时候才复制。

```
C++11 后禁止 COW：std::string 要求连续内存 + 线程安全，
COW 在并发下会出 Data Race。
```

现在所有标准库实现都用 SSO。

## 4. 常用操作复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| length()/size() | O(1) | 有 size 成员 |
| operator[] | O(1) | 随机访问，不检查边界 |
| at() | O(1) | 随机访问，越界抛异常 |
| c_str() | O(1) | 返回内部缓冲区，末尾有 \0 |
| operator+= | O(n) 均摊 | 可能触发重新分配 |
| find() | O(n) | 朴素搜索 |
| substr() | O(n) | 复制新字符串 |

## 5. string 拼接优化

```cpp
// ❌ 效率低：每次 + 创建临时对象
string r = a + b + c + d;

// ✅ 高效：一次 reserve，多次 append
string r;
r.reserve(a.size() + b.size() + c.size() + d.size());
r += a; r += b; r += c; r += d;
```

## 关键点总结

- `string` 是字符版的 dynamic array，像 vector<char>
- **SSO** 避免短字符串的堆分配，是默认实现的优化
- **COW** 已被 C++11 禁止，不再使用
- `c_str()` 保证末尾有 `\0`，可直接传给 C 函数
- 大量拼接时先 `reserve` 再 `+=`，避免反复扩容
