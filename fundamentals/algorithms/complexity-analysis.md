# Complexity Analysis - 复杂度分析

> 如何评估算法的效率？快还是慢？省内存还是费内存？

## 🎯 为什么需要复杂度分析？

**不同算法解决同一问题，效率可能差异巨大！**

```python
# 问题：判断数组中是否有重复元素

# 方法1：暴力 O(n²)
def has_duplicate_v1(nums):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False

# 方法2：排序 O(n log n)
def has_duplicate_v2(nums):
    nums.sort()
    for i in range(len(nums)-1):
        if nums[i] == nums[i+1]:
            return True
    return False

# 方法3：哈希表 O(n)
def has_duplicate_v3(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# 对于100万个元素：
# 方法1: 需要 1,000,000² = 1,000,000,000,000 次操作
# 方法2: 需要 1,000,000 × log(1,000,000) ≈ 20,000,000 次
# 方法3: 需要 1,000,000 次操作
```

---

## ⏱️ 时间复杂度 (Time Complexity)

### 什么是时间复杂度？

**算法运行时间随输入规模增长的趋势**

**注意**：
- 不是精确的运行时间
- 关注增长趋势，不关注常数
- 用大O表示法表示

---

## 📊 大O表示法 (Big O Notation)

### 常见复杂度（从快到慢）

| 复杂度 | 名称 | 例子 | n=100耗时 |
|-------|------|------|----------|
| **O(1)** | 常数 | 数组访问 | 1 |
| **O(log n)** | 对数 | 二分查找 | 7 |
| **O(n)** | 线性 | 遍历数组 | 100 |
| **O(n log n)** | 线性对数 | 快速排序 | 700 |
| **O(n²)** | 平方 | 冒泡排序 | 10,000 |
| **O(n³)** | 立方 | 三重循环 | 1,000,000 |
| **O(2ⁿ)** | 指数 | 暴力递归 | 1.27×10³⁰ |
| **O(n!)** | 阶乘 | 全排列 | 9.3×10¹⁵⁷ |

### 增长曲线

```
时间 ↑
    |                                O(n!)
    |                            ╱
    |                        ╱ O(2ⁿ)
    |                    ╱
    |                ╱ O(n³)
    |            ╱ O(n²)
    |        ╱ O(n log n)
    |    ╱ O(n)
    | ╱ O(log n)
    |___O(1)________________→ 输入规模 n
```

---

## 🔍 如何分析复杂度？

### 规则1：看循环次数

```python
# O(1) - 常数时间
def get_first(arr):
    return arr[0]  # 1次操作

# O(n) - 线性时间
def sum_array(arr):
    total = 0
    for num in arr:  # n次循环
        total += num
    return total

# O(n²) - 平方时间
def print_pairs(arr):
    for i in range(len(arr)):      # n次
        for j in range(len(arr)):  # n次
            print(arr[i], arr[j])  # n × n = n²
```

### 规则2：忽略常数

```python
# O(n)，不是O(3n)
def example(arr):
    for i in arr:      # n次
        print(i)
    for i in arr:      # n次
        print(i)
    for i in arr:      # n次
        print(i)
# 总共3n次，但记为O(n)
```

### 规则3：只看最高阶项

```python
# O(n²)，不是O(n² + n + 1)
def example(arr):
    for i in arr:                  # n次
        for j in arr:              # n²次
            print(arr[i], arr[j])
    for i in arr:                  # n次
        print(i)
    print("done")                  # 1次
# n² + n + 1 ≈ n² (当n很大时)
```

### 规则4：独立变量分开

```python
# O(m + n)，不是O(n)
def process_two_arrays(arr1, arr2):
    for i in arr1:  # m次
        print(i)
    for j in arr2:  # n次
        print(j)

# O(m × n)
def print_all_pairs(arr1, arr2):
    for i in arr1:      # m次
        for j in arr2:  # n次
            print(i, j) # m × n
```

---

## 📝 具体例子分析

### 例1：O(1) - 常数时间

```python
def is_empty(arr):
    return len(arr) == 0  # 无论数组多大，都是1次操作

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]  # 固定3次操作
```

### 例2：O(log n) - 对数时间

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1  # 每次缩小一半
        else:
            right = mid - 1  # 每次缩小一半

    return -1

# 每次循环，搜索范围减半
# n → n/2 → n/4 → ... → 1
# 需要 log₂(n) 次
```

### 例3：O(n) - 线性时间

```python
def find_max(arr):
    max_val = arr[0]
    for num in arr:  # 遍历一次，n次
        if num > max_val:
            max_val = num
    return max_val
```

### 例4：O(n log n) - 线性对数时间

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])   # 分成两半：log n 层
    right = merge_sort(arr[mid:])

    return merge(left, right)      # 每层合并：O(n)

# 总复杂度：O(n) × log(n) = O(n log n)
```

### 例5：O(n²) - 平方时间

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):           # n次
        for j in range(n-i-1):   # n-i次
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# 总次数：n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 ≈ n²/2 ≈ O(n²)
```

### 例6：O(2ⁿ) - 指数时间

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # 每次分成2个

# 调用树：
#           fib(5)
#        /          \
#     fib(4)        fib(3)
#    /     \        /     \
# fib(3) fib(2) fib(2) fib(1)
# ...
# 总节点数 ≈ 2ⁿ
```

---

## 💾 空间复杂度 (Space Complexity)

### 什么是空间复杂度？

**算法使用的额外内存随输入规模增长的趋势**

### 例子

```python
# O(1) 空间 - 常数空间
def sum_array(arr):
    total = 0  # 只用了一个变量
    for num in arr:
        total += num
    return total

# O(n) 空间 - 线性空间
def copy_array(arr):
    new_arr = []
    for num in arr:
        new_arr.append(num)  # 创建了大小为n的新数组
    return new_arr

# O(n) 空间 - 递归栈
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)  # 递归深度n，栈空间O(n)

# O(n²) 空间
def create_matrix(n):
    matrix = []
    for i in range(n):
        row = [0] * n  # n × n 的矩阵
        matrix.append(row)
    return matrix
```

---

## ⚖️ 时间与空间权衡

### 例子：斐波那契数列

```python
# 方法1：纯递归 - 时间O(2ⁿ), 空间O(n)
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

# 方法2：记忆化 - 时间O(n), 空间O(n)
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 方法3：迭代 - 时间O(n), 空间O(1)
def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# 用更多空间换取更快的时间
```

---

## 📈 最好/平均/最坏情况

### 快速排序为例

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# 最好情况：O(n log n) - 每次平分
# 平均情况：O(n log n)
# 最坏情况：O(n²) - 数组已排序，每次只分出1个元素
```

**通常我们关注最坏情况**，因为它给出了性能保证。

---

## 🎓 摊销分析 (Amortized Analysis)

### 动态数组的append

```python
# 单次操作可能是O(n)（扩容时），但平均是O(1)

class DynamicArray:
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.arr = [None] * self.capacity

    def append(self, item):
        if self.size == self.capacity:
            # 扩容：O(n)
            self._resize()
        self.arr[self.size] = item  # O(1)
        self.size += 1

    def _resize(self):
        self.capacity *= 2
        new_arr = [None] * self.capacity
        for i in range(self.size):
            new_arr[i] = self.arr[i]
        self.arr = new_arr

# 插入n个元素：
# 扩容发生在: 1, 2, 4, 8, 16, ..., n/2
# 总复制次数: 1 + 2 + 4 + 8 + ... + n/2 = n-1
# 平均每次: (n-1) / n ≈ O(1) 摊销
```

---

## 🧮 实用技巧

### 快速估算

```python
# 如果你的算法是：
# - O(1): 几乎瞬间完成
# - O(log n): 非常快
# - O(n): 可接受（n ≤ 10⁶）
# - O(n log n): 通常可接受（n ≤ 10⁶）
# - O(n²): 小数据量（n ≤ 10³）
# - O(2ⁿ): 非常小的n（n ≤ 20）
# - O(n!): 极小的n（n ≤ 11）
```

### LeetCode时间限制

```
通常1秒内：
- 10⁸ 次基本操作
- O(n) 算法：n ≤ 10⁸
- O(n log n) 算法：n ≤ 10⁶
- O(n²) 算法：n ≤ 10⁴
- O(n³) 算法：n ≤ 500
- O(2ⁿ) 算法：n ≤ 20
```

---

## 💡 优化思路

### 1. 减少循环层数

```python
# ❌ O(n³)
for i in range(n):
    for j in range(n):
        for k in range(n):
            # ...

# ✅ O(n²) - 用哈希表替代最内层循环
for i in range(n):
    for j in range(n):
        # 用哈希表 O(1) 查找
```

### 2. 提前终止

```python
# ❌ 总是遍历完整个数组
def contains(arr, target):
    found = False
    for num in arr:
        if num == target:
            found = True
    return found

# ✅ 找到就返回
def contains(arr, target):
    for num in arr:
        if num == target:
            return True  # 提前终止
    return False
```

### 3. 空间换时间

```python
# ❌ 重复计算
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)  # O(2ⁿ)

# ✅ 记忆化
memo = {}
def fib(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1) + fib(n-2)  # O(n)
    return memo[n]
```

### 4. 选择更好的数据结构

```python
# ❌ 用列表查找: O(n)
if item in my_list:
    ...

# ✅ 用集合查找: O(1)
if item in my_set:
    ...
```

---

## 🔗 相关概念

- [数据结构](../data-structures/) - 不同数据结构的复杂度
- [排序算法](sorting-searching.md) - 各种排序的复杂度对比
- [动态规划](dynamic-programming.md) - 优化重复计算

---

## 📚 复杂度速查表

| 算法/操作 | 时间复杂度 | 空间复杂度 |
|----------|-----------|-----------|
| **数组访问** | O(1) | O(1) |
| **数组搜索** | O(n) | O(1) |
| **数组插入** | O(n) | O(1) |
| **链表访问** | O(n) | O(1) |
| **链表插入** | O(1) | O(1) |
| **哈希表查找** | O(1) | O(n) |
| **BST查找** | O(log n) | O(1) |
| **堆插入** | O(log n) | O(1) |
| **冒泡排序** | O(n²) | O(1) |
| **快速排序** | O(n log n) | O(log n) |
| **归并排序** | O(n log n) | O(n) |
| **二分查找** | O(log n) | O(1) |
| **BFS** | O(V+E) | O(V) |
| **DFS** | O(V+E) | O(V) |

---

**记住**：
1. 大O表示增长趋势，不是精确时间
2. 忽略常数和低阶项
3. 通常关注最坏情况
4. 时间和空间可以互相权衡
5. 选择合适的数据结构能大幅优化复杂度
6. 实际性能还要考虑常数因子和硬件特性
