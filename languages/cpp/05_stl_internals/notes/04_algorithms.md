# STL 算法源码探秘

## 1. 算法框架

STL 算法是**基于迭代器**的操作，和容器解耦：

```cpp
// 一个算法，任意容器
template <typename InputIt, typename T>
InputIt find(InputIt first, InputIt last, const T& value) {
    while (first != last) {
        if (*first == value)
            return first;
        ++first;
    }
    return last;
}
```

## 2. sort 的实现

`std::sort` 不是单纯的快排，而是 **内省排序（IntroSort）**：

```cpp
template <typename RandomIt>
void sort(RandomIt first, RandomIt last) {
    // 1. 当递归深度 > 2*log₂(n) 时，切换到堆排序（防止 O(n²)）
    // 2. 当子数组长度 < 16 时，切换到插入排序（小数组更快）
    // 3. 默认：三数取中快排
    introsort_loop(first, last, 2 * log2(last - first));
    final_insertion_sort(first, last);  // 最后做一次插入排序收尾
}
```

**为什么混合排序？**
```
快排最坏 O(n²) → 混入堆排序保证 O(n log n)
快排小数组开销大 → 混入插入排序优化小数组
```

## 3. lower_bound / upper_bound

基于二分查找的算法，容器必须**有序**：

```cpp
// 查找第一个 >= value 的位置
template <typename ForwardIt, typename T>
ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value) {
    while (first < last) {
        auto mid = first + (last - first) / 2;
        if (*mid < value)
            first = mid + 1;
        else
            last = mid;
    }
    return first;  // O(log n)
}
```

## 4. 常用算法复杂度速查

| 算法 | 复杂度 | 备注 |
|------|--------|------|
| `find` | O(n) | 线性搜索 |
| `binary_search` | O(log n) | 要求有序 |
| `lower_bound` | O(log n) | 有序容器用 |
| `sort` | O(n log n) | 随机迭代器 |
| `stable_sort` | O(n log n) | 保持相等元素顺序 |
| `partial_sort` | O(n log k) | 只排前 k 个 |
| `nth_element` | O(n) 均摊 | 快速选择，部分排序 |
| `remove_if` | O(n) | 不删除元素，只移动 |
| `unique` | O(n) | 去重，只移动不删除 |
| `accumulate` | O(n) | 数值求和 |

## 5. remove-erase 惯用法

```cpp
// remove_if 不删除，只把符合条件的移到末尾，返回新结尾
auto it = std::remove_if(v.begin(), v.end(), [](int x){ return x < 0; });
// 真的删除
v.erase(it, v.end());

// 或者一行
v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());
```

## 关键点总结

- STL 算法通过**迭代器**操作数据，与容器解耦
- `sort` = 快排 + 堆排 + 插入排（IntroSort）保证 O(n log n)
- `lower_bound`/`upper_bound` 在**有序范围**上做二分查找
- `remove` 族算法只**移动**不删除，需配合 `erase`
- 用好 `nth_element`（O(n)）可以替代部分排序场景
