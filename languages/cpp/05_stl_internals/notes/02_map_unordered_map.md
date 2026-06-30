# map 与 unordered_map 源码对比

## 1. std::map — 红黑树

```cpp
template <typename Key, typename T, typename Compare = std::less<Key>>
class map {
    using value_type = std::pair<const Key, T>;
    // 内部是红黑树节点
    struct Node {
        value_type data;
        Node* left;
        Node* right;
        Node* parent;
        bool is_red;  // 红黑树颜色标记
    };
    Node* root_;
    size_t size_;
};
```

**红黑树特性**：
- 自平衡二叉搜索树
- 插入/删除/查找：**O(log n)**
- 元素按 key **有序排列**
- 每个节点额外存储 parent + left + right + color = 4个指针 + 1字节

## 2. std::unordered_map — 哈希表

```cpp
template <typename Key, typename T, typename Hash = std::hash<Key>>
class unordered_map {
    // 链地址法：桶数组 + 链表
    struct Node {
        std::pair<const Key, T> data;
        Node* next;  // 链表指针
    };
    Node** buckets_;   // 桶数组
    size_t bucket_count_;
    size_t size_;
    float max_load_factor_;  // 默认 1.0
};
```

**哈希表特性**：
- 插入/删除/查找：**O(1)** 均摊
- 元素**无序**
- 依赖**好用的哈希函数**
- 碰撞用链地址法解决

## 3. 性能对比

| 操作 | map (红黑树) | unordered_map (哈希表) |
|------|-------------|----------------------|
| 查找 | O(log n) | O(1) 均摊 |
| 插入 | O(log n) | O(1) 均摊 |
| 有序遍历 | O(n) 天然有序 | O(n) 输出乱序 |
| 内存 | 少（无桶） | 多（桶数组 + 链表指针） |
| 最坏情况 | O(log n) 保证 | O(n)（哈希冲突） |

**选择指南**：
- 需要有序 → map
- 需要最快查找 → unordered_map
- 数据量小（< 100）→ 差距不大
- 需要范围查询（`lower_bound`）→ map

## 4. 自定义哈希

```cpp
struct Point { int x, y; };

struct PointHash {
    size_t operator()(const Point& p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
};

std::unordered_map<Point, int, PointHash> grid;
```

## 关键点总结

- `map` = 红黑树，O(log n)，有序
- `unordered_map` = 哈希表，O(1) 均摊，无序
- 哈希表的性能依赖哈希函数质量和负载因子
- `map` 适合范围查询和有序需求
- `unordered_map` 适合高速精确查找
