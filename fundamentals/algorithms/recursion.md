# Recursion - 递归

> 函数调用自己：分解问题，解决子问题，组合答案

## 🎯 什么是递归？

**递归**是函数直接或间接调用自己的编程技术。

### 生活中的递归

1. **俄罗斯套娃** - 打开一个娃娃，里面还有更小的娃娃
2. **镜子对镜** - 无限反射
3. **定义中引用自己** - "递归：见递归"
4. **文件夹** - 文件夹里还有文件夹

---

## 🔑 递归的三要素

### 1. 基线条件 (Base Case)
**最简单的情况，直接返回结果**

### 2. 递归条件 (Recursive Case)
**将问题分解成更小的子问题**

### 3. 向基线靠近
**每次递归调用都要离基线更近**

---

## 📝 经典例子：阶乘

### 数学定义
```
n! = n × (n-1) × (n-2) × ... × 1
5! = 5 × 4 × 3 × 2 × 1 = 120

递归定义：
n! = n × (n-1)!
0! = 1  (基线条件)
```

### Python实现

```python
def factorial(n):
    # 基线条件
    if n == 0 or n == 1:
        return 1

    # 递归条件：分解成更小的问题
    return n * factorial(n - 1)

# 使用
print(factorial(5))  # 120
```

### 执行过程可视化

```
factorial(5)
= 5 * factorial(4)
= 5 * (4 * factorial(3))
= 5 * (4 * (3 * factorial(2)))
= 5 * (4 * (3 * (2 * factorial(1))))
= 5 * (4 * (3 * (2 * 1)))
= 5 * (4 * (3 * 2))
= 5 * (4 * 6)
= 5 * 24
= 120
```

### 调用栈

```
factorial(5)      ← 栈顶
  factorial(4)
    factorial(3)
      factorial(2)
        factorial(1) → 返回1
      ← 返回2
    ← 返回6
  ← 返回24
← 返回120
```

---

## 🔢 更多经典例子

### 1. 斐波那契数列

```python
def fibonacci(n):
    """F(n) = F(n-1) + F(n-2)"""
    # 基线条件
    if n <= 1:
        return n

    # 递归条件
    return fibonacci(n - 1) + fibonacci(n - 2)

# 问题：O(2^n) 时间复杂度，太慢！
```

**调用树**：
```
            fib(5)
         /          \
     fib(4)        fib(3)
    /     \        /     \
fib(3)  fib(2)  fib(2)  fib(1)
  /  \    / \     / \
...  ... ... ... ... ...

重复计算很多次！
```

**优化：记忆化递归**

```python
def fibonacci_memo(n, memo={}):
    """记忆化：O(n)"""
    # 已经计算过，直接返回
    if n in memo:
        return memo[n]

    # 基线条件
    if n <= 1:
        return n

    # 递归并保存结果
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# 或者用装饰器
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)
```

---

### 2. 求和数组

```python
def sum_array(arr):
    """递归求和"""
    # 基线条件：空数组
    if not arr:
        return 0

    # 递归条件：第一个元素 + 剩余元素的和
    return arr[0] + sum_array(arr[1:])

# 使用
print(sum_array([1, 2, 3, 4, 5]))  # 15
```

**思考过程**：
```
sum([1, 2, 3, 4, 5])
= 1 + sum([2, 3, 4, 5])
= 1 + (2 + sum([3, 4, 5]))
= 1 + (2 + (3 + sum([4, 5])))
= 1 + (2 + (3 + (4 + sum([5]))))
= 1 + (2 + (3 + (4 + (5 + sum([])))))
= 1 + (2 + (3 + (4 + (5 + 0))))
= 15
```

---

### 3. 反转字符串

```python
def reverse_string(s):
    """递归反转字符串"""
    # 基线条件
    if len(s) <= 1:
        return s

    # 递归条件：最后一个字符 + 反转前面的字符串
    return s[-1] + reverse_string(s[:-1])

# 使用
print(reverse_string("hello"))  # "olleh"
```

---

### 4. 判断回文

```python
def is_palindrome(s):
    """递归判断回文"""
    # 基线条件
    if len(s) <= 1:
        return True

    # 递归条件：首尾相等 且 中间也是回文
    if s[0] == s[-1]:
        return is_palindrome(s[1:-1])
    else:
        return False

# 使用
print(is_palindrome("racecar"))  # True
print(is_palindrome("hello"))    # False
```

---

### 5. 计算幂

```python
def power(base, exp):
    """base^exp"""
    # 基线条件
    if exp == 0:
        return 1

    # 递归条件
    return base * power(base, exp - 1)

# 优化：快速幂 O(log n)
def power_fast(base, exp):
    # 基线条件
    if exp == 0:
        return 1

    # 递归条件：利用 base^n = (base^(n/2))^2
    half = power_fast(base, exp // 2)

    if exp % 2 == 0:
        return half * half
    else:
        return base * half * half

# 2^10 = (2^5)^2 = (2 * (2^2)^2)^2
```

---

## 🌳 树的递归

### 二叉树高度

```python
def tree_height(root):
    """计算树的高度"""
    # 基线条件：空树
    if not root:
        return 0

    # 递归条件：1 + max(左子树高度, 右子树高度)
    left_height = tree_height(root.left)
    right_height = tree_height(root.right)

    return 1 + max(left_height, right_height)
```

### 二叉树遍历

```python
def preorder(root):
    """前序遍历：根-左-右"""
    if not root:
        return []

    result = [root.val]  # 根
    result += preorder(root.left)   # 左
    result += preorder(root.right)  # 右

    return result

def inorder(root):
    """中序遍历：左-根-右"""
    if not root:
        return []

    result = inorder(root.left)    # 左
    result.append(root.val)        # 根
    result += inorder(root.right)  # 右

    return result
```

### 查找二叉搜索树

```python
def search_bst(root, val):
    """在BST中查找"""
    # 基线条件
    if not root or root.val == val:
        return root

    # 递归条件：根据BST性质决定往左还是往右
    if val < root.val:
        return search_bst(root.left, val)
    else:
        return search_bst(root.right, val)
```

---

## 🔀 分治算法 (Divide and Conquer)

**将问题分成更小的子问题，解决子问题，合并结果**

### 归并排序

```python
def merge_sort(arr):
    # 基线条件
    if len(arr) <= 1:
        return arr

    # 分：分成两半
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # 治：合并两个有序数组
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result
```

### 快速排序

```python
def quick_sort(arr):
    # 基线条件
    if len(arr) <= 1:
        return arr

    # 分：选择基准，分区
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    # 治：递归排序并合并
    return quick_sort(left) + middle + quick_sort(right)
```

---

## 🔙 回溯算法 (Backtracking)

**尝试所有可能，遇到不满足条件就回退**

### 1. 全排列

```python
def permutations(nums):
    """生成所有排列"""
    result = []

    def backtrack(path, remaining):
        # 基线条件：用完所有数字
        if not remaining:
            result.append(path[:])
            return

        # 递归条件：尝试每个剩余数字
        for i in range(len(remaining)):
            # 选择
            path.append(remaining[i])
            new_remaining = remaining[:i] + remaining[i+1:]

            # 递归
            backtrack(path, new_remaining)

            # 撤销选择（回溯）
            path.pop()

    backtrack([], nums)
    return result

# 使用
print(permutations([1, 2, 3]))
# [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### 2. 子集

```python
def subsets(nums):
    """生成所有子集"""
    result = []

    def backtrack(start, path):
        # 每个路径都是一个子集
        result.append(path[:])

        # 递归条件：从start开始尝试每个数字
        for i in range(start, len(nums)):
            # 选择
            path.append(nums[i])

            # 递归
            backtrack(i + 1, path)

            # 撤销选择
            path.pop()

    backtrack(0, [])
    return result

# 使用
print(subsets([1, 2, 3]))
# [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### 3. N皇后问题

```python
def n_queens(n):
    """n皇后问题"""
    result = []
    board = [['.'] * n for _ in range(n)]

    def is_valid(row, col):
        """检查位置是否合法"""
        # 检查同一列
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # 检查左上对角线
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # 检查右上对角线
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row):
        # 基线条件：放完所有行
        if row == n:
            result.append([''.join(r) for r in board])
            return

        # 递归条件：尝试在当前行的每一列放置皇后
        for col in range(n):
            if is_valid(row, col):
                # 选择
                board[row][col] = 'Q'

                # 递归
                backtrack(row + 1)

                # 撤销选择
                board[row][col] = '.'

    backtrack(0)
    return result
```

---

## ⚠️ 递归的注意事项

### 1. 栈溢出

```python
# ❌ 深度太深会栈溢出
def infinite_recursion(n):
    return infinite_recursion(n + 1)

# Python默认递归深度约1000
import sys
print(sys.getrecursionlimit())  # 1000

# 可以设置（但不推荐）
sys.setrecursionlimit(10000)
```

### 2. 重复计算

```python
# ❌ 效率低：重复计算
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)  # O(2^n)

# ✅ 记忆化：避免重复
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n-1) + fib_memo(n-2)  # O(n)
```

### 3. 空间复杂度

```python
# 递归会占用栈空间
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

# 空间复杂度：O(n) - 递归深度
```

---

## 🔄 递归 vs 迭代

### 同一问题的两种解法

```python
# 递归版本
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)

# 迭代版本
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

### 对比

| 特性 | 递归 | 迭代 |
|-----|------|------|
| **代码** | 简洁优雅 | 相对冗长 |
| **空间** | O(n)栈空间 | O(1) |
| **速度** | 较慢（函数调用开销） | 较快 |
| **易读性** | 符合问题定义 | 需要理解循环 |
| **适用** | 树、图、分治 | 简单循环 |

---

## 💡 何时用递归？

### ✅ 适合递归的场景

1. **问题有递归结构**
   - 树和图的遍历
   - 分治问题

2. **自然的递归定义**
   - 阶乘、斐波那契
   - 汉诺塔

3. **回溯问题**
   - 排列组合
   - N皇后
   - 数独

4. **代码简洁重要**
   - 快速原型
   - 可读性优先

### ❌ 不适合递归的场景

1. **深度太深**
   - 可能栈溢出
   - 改用迭代

2. **重复计算多**
   - 没有记忆化的话很慢
   - 改用动态规划

3. **性能关键**
   - 函数调用开销大
   - 改用迭代

---

## 🔗 相关概念

- [复杂度分析](complexity-analysis.md) - 递归的时间空间复杂度
- [动态规划](dynamic-programming.md) - 记忆化递归的优化
- [树与图](../data-structures/trees-graphs.md) - 递归的经典应用

---

## 📚 经典递归题目

### 基础题目
- 阶乘
- 斐波那契
- 反转字符串
- 判断回文

### 树的递归
- 二叉树的高度
- 二叉树的遍历
- 验证二叉搜索树
- 路径总和

### 回溯
- 全排列
- 子集
- 组合
- N皇后

---

**记住**：
1. 递归 = 基线条件 + 递归条件 + 向基线靠近
2. 调用栈会占用空间
3. 记忆化避免重复计算
4. 树和图天然适合递归
5. 能用递归的也能用迭代
6. 选择取决于场景和需求
