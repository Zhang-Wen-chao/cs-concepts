"""
0-1 背包 - 0/1 Knapsack

问题：N 个物品，每个有重量 w[i] 和价值 v[i]，背包容量 W，最大价值？

状态定义：dp[i][w] = 前 i 个物品、容量 w 能装的最大价值
转移方程：dp[i][w] = max(dp[i-1][w], dp[i-1][w-w[i]] + v[i])
          ├── 不选第 i 个 → dp[i-1][w]
          └── 选第 i 个 → dp[i-1][w-w[i]] + v[i]
"""


# === 解法1：二维 DP O(N*W) / O(N*W) ===
def knapsack_2d(weights: list[int], values: list[int], capacity: int) -> int:
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i, v_i = weights[i - 1], values[i - 1]
        for w in range(1, capacity + 1):
            if w < w_i:
                # 装不下
                dp[i][w] = dp[i - 1][w]
            else:
                # 装得下 → 比较装与不装
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)

    return dp[n][capacity]


# === 解法2：一维空间压缩 O(N*W) / O(W) ===
def knapsack_1d(weights: list[int], values: list[int], capacity: int) -> int:
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # 倒序遍历，保证每个物品只用一次
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]


if __name__ == "__main__":
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    print(f"物品: weight={weights}, value={values}")
    print(f"背包容量: {capacity}")
    print(f"  二维 DP: {knapsack_2d(weights, values, capacity)}")
    print(f"  一维压缩: {knapsack_1d(weights, values, capacity)}")
    # 预期: 选 weight 3+5 = 3+4+5, value 4+6 = 10
