"""
打家劫舍 - House Robber
LeetCode 198

问题：一排房子，不能偷相邻的，求最多能偷多少？

状态定义：dp[i] = 前 i 间房子能偷的最大金额
转移方程：dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
          ├── 不偷当前 → dp[i-1]
          └── 偷当前 → dp[i-2] + nums[i-1]
"""


# === 解法1：迭代 DP O(n) / O(n) ===
def rob_dp(nums: list[int]) -> int:
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]

    dp = [0] * (n + 1)
    dp[1] = nums[0]
    for i in range(2, n + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
    return dp[n]


# === 解法2：空间压缩 O(n) / O(1) ===
def rob_optimized(nums: list[int]) -> int:
    prev, curr = 0, 0
    for num in nums:
        prev, curr = curr, max(curr, prev + num)
    return curr


if __name__ == "__main__":
    test = [2, 7, 9, 3, 1]
    print(f"房屋价值: {test}")
    print(f"  标准 DP: {rob_dp(test)}")
    print(f"  空间压缩: {rob_optimized(test)}")
    # 预期: 2 + 9 + 1 = 12
