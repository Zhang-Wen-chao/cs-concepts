# Dynamic Programming - 动态规划

> 记住过去，避免重复计算：从简单到复杂，自底向上解决问题

## 🎯 什么是动态规划？

**动态规划 (DP)** 是一种通过把原问题分解为相对简单的子问题，并存储子问题的答案来避免重复计算的算法思想。

### 核心思想

```
动态规划 = 递归 + 记忆化 + 状态转移
```

**三个要素**：
1. **重叠子问题** - 子问题会被多次计算
2. **最优子结构** - 问题的最优解包含子问题的最优解
3. **状态转移方程** - 描述子问题之间的关系

---

## 🆚 递归 vs 记忆化递归 vs 动态规划

### 问题：斐波那契数列

#### 1. 纯递归 - O(2ⁿ)

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

# 问题：大量重复计算
# fib(5) = fib(4) + fib(3)
#        = (fib(3) + fib(2)) + (fib(2) + fib(1))
#        = ((fib(2) + fib(1)) + fib(2)) + (fib(2) + fib(1))
# fib(2)被计算了3次！
```

#### 2. 记忆化递归 (自顶向下) - O(n)

```python
def fib_memo(n, memo={}):
    """从大问题往小问题递归"""
    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# 每个子问题只计算一次，保存结果
```

#### 3. 动态规划 (自底向上) - O(n)

```python
def fib_dp(n):
    """从小问题往大问题迭代"""
    if n <= 1:
        return n

    # DP数组
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1

    # 自底向上填表
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# 优化空间：只需要保存最近两个值
def fib_dp_optimized(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1
```

---

## 📋 DP的解题步骤

### 1. 定义状态

**dp[i] 表示什么？**

```python
# 例子：爬楼梯
# dp[i] = 到达第i阶的方法数
```

### 2. 找状态转移方程

**dp[i] 如何从之前的状态得到？**

```python
# 例子：爬楼梯（每次1步或2步）
# dp[i] = dp[i-1] + dp[i-2]
# 可以从i-1阶跨1步，或从i-2阶跨2步
```

### 3. 确定初始条件

**最简单的情况是什么？**

```python
# 例子：爬楼梯
# dp[0] = 1  （0阶有1种方法：不动）
# dp[1] = 1  （1阶有1种方法：跨1步）
```

### 4. 确定计算顺序

**从哪里开始计算？**

```python
# 通常从小到大
for i in range(2, n+1):
    dp[i] = dp[i-1] + dp[i-2]
```

### 5. 返回结果

```python
return dp[n]
```

---

## 💡 经典DP问题

### 1. 爬楼梯 (LeetCode 70)

**问题**：n阶楼梯，每次爬1或2步，有多少种方法？

```python
def climb_stairs(n):
    """
    状态：dp[i] = 到达第i阶的方法数
    转移：dp[i] = dp[i-1] + dp[i-2]
    初始：dp[0]=1, dp[1]=1
    """
    if n <= 1:
        return 1

    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# 空间优化
def climb_stairs_optimized(n):
    if n <= 1:
        return 1

    prev2, prev1 = 1, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1
```

---

### 2. 最小路径和 (LeetCode 64)

**问题**：从左上角到右下角，只能向右或向下，找最小路径和

```python
def min_path_sum(grid):
    """
    状态：dp[i][j] = 到达(i,j)的最小路径和
    转移：dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    初始：dp[0][0] = grid[0][0]
    """
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # 初始化第一个格子
    dp[0][0] = grid[0][0]

    # 初始化第一行（只能从左边来）
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    # 初始化第一列（只能从上边来）
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    # 填表
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]

# 空间优化：只需要一行
def min_path_sum_optimized(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n

    dp[0] = grid[0][0]
    for j in range(1, n):
        dp[j] = dp[j-1] + grid[0][j]

    for i in range(1, m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])

    return dp[n-1]
```

---

### 3. 最长递增子序列 (LeetCode 300)

**问题**：找数组中最长递增子序列的长度

```python
def length_of_LIS(nums):
    """
    状态：dp[i] = 以nums[i]结尾的最长递增子序列长度
    转移：dp[i] = max(dp[j] + 1) for all j < i if nums[j] < nums[i]
    初始：dp[i] = 1（每个元素本身是长度为1的序列）
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n  # 初始化为1

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)  # 返回最大值

# 例子：[10, 9, 2, 5, 3, 7, 101, 18]
# dp = [1,  1, 1, 2, 2, 3,  4,   4]
#           ↑     ↑     ↑   ↑
#          nums[2]<nums[3] → dp[3]=dp[2]+1=2
#          nums[3]<nums[5] → dp[5]=dp[3]+1=3
#          nums[5]<nums[6] → dp[6]=dp[5]+1=4
```

---

### 4. 0-1背包问题

**问题**：n个物品，每个有重量和价值，背包容量W，求最大价值

```python
def knapsack(weights, values, W):
    """
    状态：dp[i][w] = 前i个物品，背包容量w时的最大价值
    转移：
      不选第i个：dp[i][w] = dp[i-1][w]
      选第i个：  dp[i][w] = dp[i-1][w-weights[i]] + values[i]
      取最大：    dp[i][w] = max(不选, 选)
    """
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            # 不选第i个物品
            dp[i][w] = dp[i-1][w]

            # 如果能放下第i个物品，考虑选它
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w],
                              dp[i-1][w-weights[i-1]] + values[i-1])

    return dp[n][W]

# 空间优化：一维数组（注意倒序）
def knapsack_optimized(weights, values, W):
    dp = [0] * (W + 1)

    for i in range(len(weights)):
        # 倒序！避免重复使用同一物品
        for w in range(W, weights[i]-1, -1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])

    return dp[W]
```

---

### 5. 最长公共子序列 (LCS)

**问题**：两个字符串的最长公共子序列

```python
def longest_common_subsequence(text1, text2):
    """
    状态：dp[i][j] = text1[0:i]和text2[0:j]的LCS长度
    转移：
      如果text1[i-1] == text2[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
      否则:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# 例子："abcde" 和 "ace"
#     ""  a  c  e
#  "" 0  0  0  0
#  a  0  1  1  1
#  b  0  1  1  1
#  c  0  1  2  2
#  d  0  1  2  2
#  e  0  1  2  3  ← 答案
```

---

### 6. 编辑距离 (LeetCode 72)

**问题**：将word1转换为word2的最少操作数（插入、删除、替换）

```python
def min_distance(word1, word2):
    """
    状态：dp[i][j] = word1[0:i]转换为word2[0:j]的最少操作数
    转移：
      如果word1[i-1] == word2[j-1]:
        dp[i][j] = dp[i-1][j-1]  （不需要操作）
      否则：
        dp[i][j] = 1 + min(
          dp[i-1][j],    # 删除
          dp[i][j-1],    # 插入
          dp[i-1][j-1]   # 替换
        )
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化：空串到其他串的距离
    for i in range(m + 1):
        dp[i][0] = i  # 需要i次删除
    for j in range(n + 1):
        dp[0][j] = j  # 需要j次插入

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # 删除
                    dp[i][j-1],    # 插入
                    dp[i-1][j-1]   # 替换
                )

    return dp[m][n]
```

---

### 7. 打家劫舍 (LeetCode 198)

**问题**：不能抢劫相邻的房子，求最大金额

```python
def rob(nums):
    """
    状态：dp[i] = 抢劫前i个房子的最大金额
    转移：dp[i] = max(dp[i-1], dp[i-2] + nums[i])
          不抢第i个    抢第i个（不能抢i-1）
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    return dp[n-1]

# 空间优化
def rob_optimized(nums):
    if not nums:
        return 0

    prev2, prev1 = 0, 0
    for num in nums:
        current = max(prev1, prev2 + num)
        prev2, prev1 = prev1, current

    return prev1
```

---

### 8. 零钱兑换 (LeetCode 322)

**问题**：用最少的硬币凑出金额

```python
def coin_change(coins, amount):
    """
    状态：dp[i] = 凑出金额i需要的最少硬币数
    转移：dp[i] = min(dp[i-coin] + 1) for coin in coins
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 金额0需要0个硬币

    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i-coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# 例子：coins=[1,2,5], amount=11
# dp[0]=0
# dp[1]=1  (1个硬币1)
# dp[2]=1  (1个硬币2)
# dp[3]=2  (1+1 或 2+1)
# ...
# dp[11]=3 (5+5+1)
```

---

## 🎯 DP的类型

### 1. 线性DP
一维或二维数组，线性扫描

**例子**：爬楼梯、打家劫舍、最长递增子序列

### 2. 区间DP
在区间上进行DP

**例子**：最长回文子串

### 3. 背包DP
背包问题及其变体

**例子**：0-1背包、完全背包、多重背包

### 4. 树形DP
在树上进行DP

**例子**：树的直径、打家劫舍III

### 5. 状态压缩DP
用位运算压缩状态

**例子**：旅行商问题(TSP)

---

## 💡 DP优化技巧

### 1. 空间优化

```python
# 原始：二维数组
dp = [[0] * n for _ in range(m)]

# 优化：一维数组（如果只需要上一行）
dp = [0] * n

# 优化：常数空间（如果只需要几个变量）
prev2, prev1 = 0, 1
```

### 2. 滚动数组

```python
# 只保留最近的k行
dp = [[0] * n for _ in range(k)]
dp[i % k][j] = ...  # 用取模实现滚动
```

### 3. 状态压缩

```python
# 用位运算表示集合
state = (1 << n) - 1  # 所有位都是1
if state & (1 << i):  # 检查第i位
    state ^= (1 << i)  # 翻转第i位
```

---

## 🔗 相关概念

- [递归](recursion.md) - DP是递归+记忆化
- [复杂度分析](complexity-analysis.md) - 分析DP的时间空间复杂度
- [贪心算法](../algorithms/) - 有时贪心比DP更优

---

## 📚 DP学习路线

### 入门题目
1. 爬楼梯 (LeetCode 70)
2. 打家劫舍 (LeetCode 198)
3. 最大子数组和 (LeetCode 53)

### 进阶题目
4. 零钱兑换 (LeetCode 322)
5. 最长递增子序列 (LeetCode 300)
6. 最长公共子序列 (LeetCode 1143)

### 困难题目
7. 编辑距离 (LeetCode 72)
8. 正则表达式匹配 (LeetCode 10)
9. 最长回文子串 (LeetCode 5)

---

## 💭 DP思维训练

### 如何识别DP问题？

✅ **适合DP的信号**：
- 问"最大/最小/最长/最多"
- 问"有多少种方法"
- 有明显的重叠子问题
- 可以分解成子问题

❌ **不适合DP**：
- 需要输出所有方案（用回溯）
- 贪心可以解决
- 没有重叠子问题

### DP vs 其他方法

```
问题：硬币找零

贪心：每次选最大面值
- 优点：快 O(n)
- 缺点：不一定最优

DP：考虑所有可能
- 优点：保证最优
- 缺点：慢 O(amount × coins)

选择：看是否有贪心选择性质
```

---

**记住**：
1. DP = 递归 + 记忆化 + 状态转移
2. 五个步骤：定义状态、转移方程、初始条件、计算顺序、返回结果
3. 重叠子问题 + 最优子结构 = DP
4. 自顶向下（记忆化递归）或自底向上（DP数组）
5. 优化：空间压缩、滚动数组
6. 多做题，培养DP直觉
