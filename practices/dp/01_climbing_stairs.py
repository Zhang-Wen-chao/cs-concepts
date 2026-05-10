"""
爬楼梯 - Climbing Stairs
LeetCode 70

问题：每次可以爬 1 或 2 阶，到第 n 阶有多少种方法？

递归定义：dp[i] = dp[i-1] + dp[i-2]
本质：斐波那契数列
"""


# === 解法1：暴力递归 O(2^n) ===
def climb_stairs_recursive(n: int) -> int:
    if n <= 2:
        return n
    return climb_stairs_recursive(n - 1) + climb_stairs_recursive(n - 2)


# === 解法2：记忆化递归 (Top-Down DP) O(n) ===
def climb_stairs_memo(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    if n <= 2:
        return n
    if n not in memo:
        memo[n] = climb_stairs_memo(n - 1, memo) + climb_stairs_memo(n - 2, memo)
    return memo[n]


# === 解法3：迭代 DP (Bottom-Up) O(n) / O(n) ===
def climb_stairs_dp(n: int) -> int:
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


# === 解法4：空间压缩 O(n) / O(1) ===
def climb_stairs_optimized(n: int) -> int:
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


if __name__ == "__main__":
    n = 10
    print(f"n = {n}")
    print(f"  暴力递归: {climb_stairs_recursive(n)}")
    print(f"  记忆化递归: {climb_stairs_memo(n)}")
    print(f"  迭代 DP: {climb_stairs_dp(n)}")
    print(f"  空间压缩: {climb_stairs_optimized(n)}")
