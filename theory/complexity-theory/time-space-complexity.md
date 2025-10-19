# 时间与空间复杂度 (Time and Space Complexity)

> 衡量算法效率的标尺

## 🎯 核心思想

复杂度分析回答一个问题：**算法需要多少资源？**

**一句话理解：**
- 时间复杂度 = 运行需要多少步
- 空间复杂度 = 运行需要多少内存

**为什么重要：**
```
同样的问题，不同算法效率差异巨大：

查找问题（在n个元素中找一个）：
• 线性查找：O(n)          1000个元素 → 1000步
• 二分查找：O(log n)      1000个元素 → 10步

排序问题（给n个元素排序）：
• 冒泡排序：O(n²)         1000个元素 → 1,000,000步
• 快速排序：O(n log n)    1000个元素 → 10,000步

差距100倍！
```

## 📖 时间复杂度

### 1. 什么是时间复杂度？

#### 直观理解

```
时间复杂度 = 算法执行的"基本操作"次数（关于输入规模n）

不是：
✗ 实际运行的秒数（和硬件相关）
✗ 代码的行数

而是：
✓ 操作次数随输入增长的趋势
✓ 与具体常数无关的"增长率"
```

#### 例子

```python
# 例1：常数时间
def get_first(arr):
    return arr[0]        # 1次操作

时间复杂度：O(1)
无论数组多大，都只需1步

# 例2：线性时间
def sum_array(arr):
    total = 0
    for x in arr:        # n次循环
        total += x       # 每次1次操作
    return total

时间复杂度：O(n)
n个元素 → n次加法

# 例3：平方时间
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):       # n次外循环
        for j in range(n):   # n次内循环
            if arr[i] < arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

时间复杂度：O(n²)
n × n = n²次比较

# 例4：对数时间
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

时间复杂度：O(log n)
每次减半搜索空间
```

### 2. 大O记号 (Big-O Notation)

#### 定义

```
f(n) = O(g(n)) 表示：
当n足够大时，f(n) ≤ c × g(n)（存在常数c）

直观理解：
• f(n)的增长速度不超过g(n)
• 忽略常数因子和低阶项

例子：
3n² + 5n + 10 = O(n²)
因为：当n很大时，n²项主导一切
```

#### 为什么忽略常数？

```
实际运行时间 = c₁ × n² + c₂ × n + c₃

常数c₁, c₂, c₃取决于：
• 硬件速度
• 编译器优化
• 操作系统
• 编程语言

复杂度分析关注：
• 算法本质的增长趋势
• 与具体实现无关的特性

当n很大时：
• 2n²和1000n²增长趋势相同（都是O(n²)）
• 但n²和n log n增长趋势不同
```

### 3. 常见复杂度类

```
从快到慢：

O(1)         常数时间
O(log n)     对数时间
O(n)         线性时间
O(n log n)   线性对数时间
O(n²)        平方时间
O(n³)        立方时间
O(2ⁿ)        指数时间
O(n!)        阶乘时间

可视化（n=1000）：
O(1):        1步
O(log n):    10步
O(n):        1,000步
O(n log n):  10,000步
O(n²):       1,000,000步
O(2ⁿ):       2¹⁰⁰⁰步（宇宙原子总数的级别）
O(n!):       无法计算
```

#### O(1) - 常数时间

```python
# 数组访问
def access(arr, i):
    return arr[i]

# 哈希表查询
def lookup(hash_map, key):
    return hash_map[key]

# 栈操作
def push(stack, item):
    stack.append(item)

特点：
• 操作次数与输入规模无关
• 最理想的复杂度
```

#### O(log n) - 对数时间

```python
# 二分查找
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 为什么是O(log n)？
每次循环，搜索空间减半：
n → n/2 → n/4 → n/8 → ... → 1
需要多少步？log₂(n)步

# 平衡二叉树操作
def tree_search(root, target):
    # 每次选择一个子树
    # 树高 = O(log n)
    pass

例子：
n = 1024
log₂(1024) = 10
只需10步！
```

#### O(n) - 线性时间

```python
# 遍历数组
def find_max(arr):
    max_val = arr[0]
    for x in arr:           # n次
        if x > max_val:
            max_val = x
    return max_val

# 链表遍历
def count_nodes(head):
    count = 0
    current = head
    while current:          # n次
        count += 1
        current = current.next
    return count

特点：
• 每个元素处理一次
• 常见且高效
```

#### O(n log n) - 线性对数时间

```python
# 归并排序
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])      # T(n/2)
    right = merge_sort(arr[mid:])     # T(n/2)
    return merge(left, right)         # O(n)

# 递归树：
#       n             1层，每层O(n)工作
#      / \
#    n/2 n/2          2层，每层O(n)工作
#    / \ / \
#  n/4...  ...        4层，每层O(n)工作
#  ...
# 总共log(n)层，每层O(n) → O(n log n)

# 快速排序（平均）
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)

特点：
• 最优比较排序的复杂度
• 分治算法常见复杂度
```

#### O(n²) - 平方时间

```python
# 冒泡排序
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):              # n次
        for j in range(n - i - 1):  # n次
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# 选择排序
def selection_sort(arr):
    for i in range(len(arr)):       # n次
        min_idx = i
        for j in range(i+1, len(arr)):  # n次
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# 嵌套循环
def print_pairs(arr):
    for i in arr:                   # n次
        for j in arr:               # n次
            print(i, j)             # n²次输出

特点：
• 嵌套循环
• 小数据可接受，大数据慢
```

#### O(2ⁿ) - 指数时间

```python
# 斐波那契（朴素递归）
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

# 递归树：
#           fib(5)
#          /      \
#      fib(4)    fib(3)
#      /   \      /   \
#   fib(3) fib(2) ...
#   ...

# 节点数：2⁰ + 2¹ + 2² + ... + 2ⁿ ≈ 2ⁿ

# 子集生成
def subsets(arr):
    if not arr:
        return [[]]
    first = arr[0]
    rest = subsets(arr[1:])
    return rest + [[first] + s for s in rest]

# 每个元素：选或不选 → 2ⁿ种组合

特点：
• 非常慢！
• n=30就很难算
• 通常需要优化
```

#### O(n!) - 阶乘时间

```python
# 全排列
def permutations(arr):
    if len(arr) <= 1:
        return [arr]
    result = []
    for i in range(len(arr)):
        rest = arr[:i] + arr[i+1:]
        for p in permutations(rest):
            result.append([arr[i]] + p)
    return result

# 旅行商问题（暴力）
def tsp_brute_force(cities):
    # 尝试所有可能的路径
    # n个城市 → n!种路径
    pass

特点：
• 极慢！
• n=10 → 3,628,800
• n=20 → 2.4 × 10¹⁸（几千年）
```

### 4. 复杂度分析技巧

#### 循环规则

```python
# 单循环：O(n)
for i in range(n):
    print(i)

# 嵌套循环：相乘
for i in range(n):        # O(n)
    for j in range(m):    # O(m)
        print(i, j)       # O(n × m)

# 顺序语句：相加
for i in range(n):        # O(n)
    print(i)
for j in range(m):        # O(m)
    print(j)
# 总共：O(n + m) = O(max(n, m))

# 递减循环
for i in range(n, 0, -1):     # n次
    for j in range(i):        # i次
        print(i, j)
# 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n²)
```

#### 递归分析

```python
# 递归关系 → 主定理

# 例1：二分查找
def binary_search(arr, target):
    if len(arr) == 0:
        return -1
    mid = len(arr) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr[mid+1:], target)
    else:
        return binary_search(arr[:mid], target)

# 递归式：T(n) = T(n/2) + O(1)
# 解：T(n) = O(log n)

# 例2：归并排序
# T(n) = 2T(n/2) + O(n)
# 解：T(n) = O(n log n)

# 例3：斐波那契（朴素）
# T(n) = T(n-1) + T(n-2) + O(1)
# 解：T(n) = O(2ⁿ)（接近指数）
```

#### 主定理 (Master Theorem)

```
对于递归式：T(n) = aT(n/b) + f(n)

a：子问题个数
b：子问题规模缩小倍数
f(n)：合并代价

情况1：f(n) = O(n^c)，其中c < log_b(a)
     → T(n) = O(n^(log_b(a)))

情况2：f(n) = O(n^c log^k(n))，其中c = log_b(a)
     → T(n) = O(n^c log^(k+1)(n))

情况3：f(n) = O(n^c)，其中c > log_b(a)
     → T(n) = O(f(n))

例子：
T(n) = 2T(n/2) + O(n)
a=2, b=2, c=1
log₂(2) = 1 = c → 情况2
T(n) = O(n log n)
```

### 5. 最好、最坏、平均情况

```python
# 快速排序
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)

最好情况：O(n log n)
  • 每次pivot正好在中间
  • 完美分割

最坏情况：O(n²)
  • 每次pivot是最小/最大值
  • 如：已排序数组，选第一个为pivot
  • 退化为冒泡排序

平均情况：O(n log n)
  • 随机输入
  • pivot在中间附近的概率大

实践：
• 随机选择pivot（避免最坏情况）
• 或使用三数取中
```

```python
# 线性查找
def linear_search(arr, target):
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1

最好情况：O(1)
  • target在第一个位置

最坏情况：O(n)
  • target在最后或不存在

平均情况：O(n)
  • 平均需要检查n/2个元素
  • 仍是O(n)
```

## 📖 空间复杂度

### 1. 什么是空间复杂度？

```
空间复杂度 = 算法使用的额外内存（关于输入规模n）

包括：
✓ 动态分配的内存
✓ 递归调用栈
✓ 临时变量

不包括：
✗ 输入本身占用的空间
✗ 输出占用的空间
```

### 2. 常见空间复杂度

#### O(1) - 常数空间

```python
# 原地交换
def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
    # 只用了几个变量

# 迭代求和
def sum_iterative(arr):
    total = 0      # O(1)空间
    for x in arr:
        total += x
    return total

# 原地反转
def reverse_in_place(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
```

#### O(n) - 线性空间

```python
# 复制数组
def copy_array(arr):
    return arr[:]    # 新数组O(n)空间

# 归并排序
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])   # 递归深度O(log n)
    right = merge_sort(arr[mid:])  # 但每层合并需要O(n)
    return merge(left, right)      # 临时数组O(n)

# 总空间：O(n)

# 哈希表
def count_frequency(arr):
    freq = {}           # O(n)空间
    for x in arr:
        freq[x] = freq.get(x, 0) + 1
    return freq
```

#### O(log n) - 对数空间

```python
# 二分查找（递归）
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid+1, right)
    else:
        return binary_search_recursive(arr, target, left, mid-1)

# 递归深度：O(log n)
# 每次调用栈帧：O(1)
# 总空间：O(log n)

# 快速排序（原地，平均）
# 递归深度：O(log n)
# 每次调用：O(1)
# 总空间：O(log n)
```

### 3. 时间-空间权衡

```python
# 斐波那契数列

# 版本1：指数时间，常数空间
def fib_slow(n):
    if n <= 1:
        return n
    return fib_slow(n-1) + fib_slow(n-2)
# 时间：O(2ⁿ)，空间：O(n)（递归栈）

# 版本2：线性时间，线性空间（动态规划）
def fib_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
# 时间：O(n)，空间：O(n)

# 版本3：线性时间，常数空间（优化）
def fib_optimal(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
# 时间：O(n)，空间：O(1)

权衡：
• 版本1：简单但极慢
• 版本2：快但用更多内存
• 版本3：最优解
```

## 🔧 实际应用

### 1. 选择数据结构

```
操作             数组    链表    哈希表   平衡树
─────────────────────────────────────────────
访问第i个元素    O(1)    O(n)    N/A     O(log n)
搜索元素         O(n)    O(n)    O(1)    O(log n)
插入（开头）     O(n)    O(1)    O(1)    O(log n)
插入（末尾）     O(1)*   O(1)    O(1)    O(log n)
删除             O(n)    O(n)    O(1)    O(log n)

*假设不需要扩容

选择依据：
• 频繁随机访问 → 数组
• 频繁插入删除 → 链表
• 需要快速查找 → 哈希表
• 需要有序+查找 → 平衡树
```

### 2. 算法选择

```
问题：排序n个元素

小数据（n < 100）：
• 插入排序 O(n²) - 简单，常数小

中等数据（100 < n < 10000）：
• 快速排序 O(n log n) - 平均最快

大数据（n > 10000）：
• 归并排序 O(n log n) - 稳定，可预测

特殊情况：
• 几乎有序 → 插入排序 O(n)
• 范围小（如年龄） → 计数排序 O(n+k)
• 字符串 → 基数排序 O(nk)
```

### 3. 性能估算

```
实际运行时间估算：

假设：现代计算机每秒10⁸次操作

n = 10⁶（100万）：
O(log n):     20次操作           ≈ 瞬间
O(n):         10⁶次操作          ≈ 0.01秒
O(n log n):   2×10⁷次操作        ≈ 0.2秒
O(n²):        10¹²次操作         ≈ 3小时
O(2ⁿ):        无法计算           ≈ 宇宙年龄

实践准则：
• O(1), O(log n), O(n) → 任何规模都OK
• O(n log n) → n < 10⁷可接受
• O(n²) → n < 10⁴可接受
• O(2ⁿ) → n < 25才能算
• O(n!) → n < 11才能算
```

## 🔗 与其他概念的联系

### 与数据结构
- **数组** - O(1)访问，O(n)搜索
- **链表** - O(1)插入，O(n)访问
- **树** - O(log n)操作（平衡时）
- **哈希表** - O(1)平均查找

参考：`fundamentals/data-structures/`

### 与算法
- **排序** - 比较排序下界 Ω(n log n)
- **查找** - 二分查找 O(log n)
- **动态规划** - 时空权衡

参考：`fundamentals/algorithms/`

### 与可计算性
- **可计算性** - 能不能算
- **复杂度** - 多快能算
- P vs NP - 可计算但可能很慢

参考：`theory/computation-theory/computability.md`

## 📖 扩展阅读

### 高级主题
- 摊还分析 (Amortized Analysis)
- 随机算法复杂度
- 并行算法复杂度
- 缓存复杂度模型

### 证明技术
- 递归树方法
- 替换法
- 主定理
- 势能法

---

**掌握复杂度分析，你就能评估和优化算法！** ⏱️
