# DP Practice - 动态规划习题集

经典 DP 题目的 Python 实现，配套 [dynamic-programming.md](../../fundamentals/algorithms/dynamic-programming.md) 的概念理解。

## 题目列表

| # | 题目 | 类型 | 状态定义 |
|---|------|------|---------|
| 01 | [爬楼梯](01_climbing_stairs.py) | 一维 DP | `dp[i]` = 到第 i 阶的方法数 |
| 02 | [打家劫舍](02_house_robber.py) | 一维 DP | `dp[i]` = 前 i 间能偷的最大值 |
| 03 | [0-1 背包](03_knapsack_01.py) | 二维/一维 | `dp[w]` = 容量 w 能装的最大价值 |
| 04 | [最长递增子序列](04_longest_increasing_subsequence.py) | 一维 DP | `dp[i]` = 以 i 结尾的 LIS 长度 |
| 05 | [最长公共子序列](05_longest_common_subsequence.py) | 二维 DP | `dp[i][j]` = s1[:i] 与 s2[:j] 的 LCS 长度 |
| 06 | [编辑距离](06_edit_distance.py) | 二维 DP | `dp[i][j]` = s1[:i] → s2[:j] 的最小编辑次数 |

## 导航

- 每个文件独立可运行：`python <文件名>`
- 同时包含递归 → 记忆化 → DP 的进化路线
- 空间压缩版本单独标注
