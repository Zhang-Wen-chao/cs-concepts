# Sorting and Searching - 排序与搜索

> 最基本、最常用的算法：如何整理数据？如何快速查找？

## 🔍 搜索算法

### 1. 线性搜索 (Linear Search)

**从头到尾逐个查找**

```python
def linear_search(arr, target):
    """O(n)时间，O(1)空间"""
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # 返回索引
    return -1  # 未找到

# 使用
arr = [3, 5, 2, 4, 9]
print(linear_search(arr, 4))  # 3
print(linear_search(arr, 7))  # -1
```

**特点**：
- ✅ 简单直接
- ✅ 适用于无序数组
- ❌ 效率低 O(n)

---

### 2. 二分搜索 (Binary Search)

**在有序数组中折半查找**

```python
def binary_search(arr, target):
    """O(log n)时间，O(1)空间"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1  # 在右半边
        else:
            right = mid - 1  # 在左半边

    return -1

# 使用（数组必须有序）
arr = [1, 3, 5, 7, 9, 11, 13]
print(binary_search(arr, 7))   # 3
print(binary_search(arr, 10))  # -1
```

**过程可视化**：
```
查找 7:
[1, 3, 5, 7, 9, 11, 13]
          ↑ mid=5, 5<7, 去右边

[7, 9, 11, 13]
 ↑ mid=7, 找到！
```

**递归版本**：
```python
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

**特点**：
- ✅ 非常快 O(log n)
- ❌ 需要有序数组
- ❌ 需要随机访问（数组）

**变体：查找边界**

```python
def find_first(arr, target):
    """找第一个等于target的位置"""
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # 继续向左找
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

def find_last(arr, target):
    """找最后一个等于target的位置"""
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1  # 继续向右找
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# 使用
arr = [1, 2, 2, 2, 3, 4]
print(find_first(arr, 2))  # 1
print(find_last(arr, 2))   # 3
```

---

## 📊 排序算法

### 排序算法对比

| 算法 | 平均时间 | 最坏时间 | 空间 | 稳定性 |
|-----|---------|---------|------|--------|
| **冒泡排序** | O(n²) | O(n²) | O(1) | ✅ |
| **选择排序** | O(n²) | O(n²) | O(1) | ❌ |
| **插入排序** | O(n²) | O(n²) | O(1) | ✅ |
| **快速排序** | O(n log n) | O(n²) | O(log n) | ❌ |
| **归并排序** | O(n log n) | O(n log n) | O(n) | ✅ |
| **堆排序** | O(n log n) | O(n log n) | O(1) | ❌ |

**稳定性**：相等元素排序后相对顺序不变

---

### 1. 冒泡排序 (Bubble Sort)

**相邻元素比较，大的往后冒泡**

```python
def bubble_sort(arr):
    """O(n²)时间，O(1)空间"""
    n = len(arr)

    for i in range(n):
        # 每轮把最大的冒泡到末尾
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    return arr

# 优化版本：提前终止
def bubble_sort_optimized(arr):
    n = len(arr)

    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:  # 如果这轮没有交换，说明已排序
            break

    return arr

# 使用
arr = [64, 34, 25, 12, 22, 11, 90]
print(bubble_sort(arr.copy()))
```

**过程可视化**：
```
[64, 34, 25, 12, 22, 11, 90]
第1轮：大的往后冒
[34, 25, 12, 22, 11, 64, 90]
第2轮：
[25, 12, 22, 11, 34, 64, 90]
...
```

---

### 2. 选择排序 (Selection Sort)

**每次选择最小的放到前面**

```python
def selection_sort(arr):
    """O(n²)时间，O(1)空间"""
    n = len(arr)

    for i in range(n):
        # 找到未排序部分的最小值
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # 交换到正确位置
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr
```

**过程可视化**：
```
[64, 25, 12, 22, 11]
找最小(11)，交换到第0位
[11, 25, 12, 22, 64]
找剩余最小(12)，交换到第1位
[11, 12, 25, 22, 64]
...
```

---

### 3. 插入排序 (Insertion Sort)

**像整理扑克牌，逐个插入到正确位置**

```python
def insertion_sort(arr):
    """O(n²)时间，O(1)空间"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        # 把key插入到前面已排序部分的正确位置
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]  # 向后移
            j -= 1

        arr[j + 1] = key

    return arr
```

**过程可视化**：
```
[5, 2, 4, 6, 1, 3]
已排序: [5] | 待排序: [2, 4, 6, 1, 3]
插入2: [2, 5] | [4, 6, 1, 3]
插入4: [2, 4, 5] | [6, 1, 3]
插入6: [2, 4, 5, 6] | [1, 3]
插入1: [1, 2, 4, 5, 6] | [3]
插入3: [1, 2, 3, 4, 5, 6]
```

**特点**：
- ✅ 对几乎已排序的数组很快
- ✅ 稳定排序
- ❌ 大数据集慢

---

### 4. 快速排序 (Quick Sort)

**选择基准(pivot)，分区，递归排序**

```python
def quick_sort(arr):
    """O(n log n)平均，O(n²)最坏"""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准

    # 分成三部分
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

# 原地排序版本
def quick_sort_inplace(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pi = partition(arr, low, high)
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)

    return arr

def partition(arr, low, high):
    """分区函数"""
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

**过程可视化**：
```
[3, 6, 8, 10, 1, 2, 1]
选pivot=10
分区: [3, 6, 8, 1, 2, 1] < 10 < []
递归左边...
```

**特点**：
- ✅ 平均O(n log n)，很快
- ✅ 原地排序
- ❌ 最坏O(n²)（已排序）
- ❌ 不稳定

---

### 5. 归并排序 (Merge Sort)

**分治：分成两半，递归排序，合并**

```python
def merge_sort(arr):
    """O(n log n)时间，O(n)空间"""
    if len(arr) <= 1:
        return arr

    # 分
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # 合并
    return merge(left, right)

def merge(left, right):
    """合并两个有序数组"""
    result = []
    i = j = 0

    # 比较并合并
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # 添加剩余元素
    result.extend(left[i:])
    result.extend(right[j:])

    return result
```

**过程可视化**：
```
[38, 27, 43, 3, 9, 82, 10]

分：
        [38,27,43,3,9,82,10]
       /                    \
  [38,27,43,3]          [9,82,10]
   /        \            /      \
[38,27]  [43,3]      [9,82]    [10]
  /  \    /  \        /  \
[38][27][43][3]     [9][82]

合并：
[27,38] [3,43]     [9,82] [10]
   \      /          \      /
  [3,27,38,43]     [9,10,82]
        \              /
    [3,9,10,27,38,43,82]
```

**特点**：
- ✅ 稳定O(n log n)
- ✅ 稳定排序
- ❌ 需要O(n)额外空间

---

### 6. 堆排序 (Heap Sort)

**用最大堆，逐个取出最大值**

```python
def heap_sort(arr):
    """O(n log n)时间，O(1)空间"""
    n = len(arr)

    # 建立最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 逐个取出最大值
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # 交换
        heapify(arr, i, 0)  # 重新调整堆

    return arr

def heapify(arr, n, i):
    """调整堆"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

**特点**：
- ✅ O(n log n)保证
- ✅ 原地排序
- ❌ 不稳定
- ❌ 常数因子大，实际比快排慢

---

## 🎯 选择排序算法

### 根据数据特点

```python
# 小数据（n < 50）：插入排序
if len(arr) < 50:
    return insertion_sort(arr)

# 几乎已排序：插入排序
# 完全随机：快速排序
# 需要稳定性：归并排序
# 内存受限：堆排序

# Python内置排序（Timsort）：
# 结合归并和插入，适应不同场景
arr.sort()  # O(n log n)
```

### Python的sorted和sort

```python
# sort(): 原地排序
arr = [3, 1, 4, 1, 5]
arr.sort()
print(arr)  # [1, 1, 3, 4, 5]

# sorted(): 返回新列表
arr = [3, 1, 4, 1, 5]
new_arr = sorted(arr)
print(arr)      # [3, 1, 4, 1, 5] 不变
print(new_arr)  # [1, 1, 3, 4, 5]

# 自定义排序
arr = [(1, 'c'), (2, 'a'), (3, 'b')]
arr.sort(key=lambda x: x[1])  # 按第二个元素排序
# [(2, 'a'), (3, 'b'), (1, 'c')]

# 降序
arr.sort(reverse=True)
```

---

## 💡 实战应用

### 1. Top K 问题

```python
# 方法1：排序 O(n log n)
def top_k_sort(arr, k):
    return sorted(arr, reverse=True)[:k]

# 方法2：堆 O(n log k) - 更好
import heapq

def top_k_heap(arr, k):
    # 维护大小为k的最小堆
    heap = []
    for num in arr:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap

# 方法3：快速选择 O(n)平均 - 最好
def quick_select(arr, k):
    if len(arr) == 1:
        return arr[0]

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x > pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x < pivot]

    if k <= len(left):
        return quick_select(left, k)
    elif k <= len(left) + len(mid):
        return mid[0]
    else:
        return quick_select(right, k - len(left) - len(mid))
```

### 2. 合并有序数组

```python
def merge_sorted_arrays(arr1, arr2):
    """O(n+m)时间，O(n+m)空间"""
    result = []
    i = j = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    result.extend(arr1[i:])
    result.extend(arr2[j:])

    return result
```

### 3. 查找旋转数组中的最小值

```python
def find_min_rotated(arr):
    """O(log n)"""
    left, right = 0, len(arr) - 1

    while left < right:
        mid = (left + right) // 2

        if arr[mid] > arr[right]:
            left = mid + 1  # 最小值在右边
        else:
            right = mid  # 最小值在左边或就是mid

    return arr[left]

# 例子: [4,5,6,7,0,1,2] → 0
```

---

## 🔗 相关概念

- [复杂度分析](complexity-analysis.md) - 评估排序算法效率
- [数据结构](../data-structures/) - 堆用于堆排序
- [递归](recursion.md) - 快排和归并都用递归

---

**记住**：
1. 二分搜索需要有序数组，O(log n)
2. 简单排序O(n²)：冒泡、选择、插入
3. 高效排序O(n log n)：快排、归并、堆排序
4. Python的sort是Timsort，O(n log n)
5. Top K问题用堆，O(n log k)
6. 实际项目优先用内置排序
