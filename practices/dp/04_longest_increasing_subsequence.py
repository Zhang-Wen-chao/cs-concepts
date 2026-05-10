"""
最长递增子序列 - Longest Increasing Subsequence (LIS)
LeetCode 300

问题：无序整数数组，找最长的严格递增子序列长度。
注意：子序列不要求连续，但保持原顺序。

状态定义：dp[i] = 以 nums[i] 结尾的 LIS 长度
转移方程：dp[i] = max(dp[j] + 1) for j < i and nums[j] < nums[i]
"""


# === 解法1：标准 DP O(n²) / O(n) ===
def length_of_lis(nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


# === 解法2：贪心 + 二分 O(n log n) ===
def length_of_lis_optimized(nums: list[int]) -> int:
    """
    维护 tails 数组：
    tails[i] = 长度为 i+1 的递增子序列的最小末尾值
    """
    import bisect

    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


if __name__ == "__main__":
    test = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"数组: {test}")
    print(f"  标准 DP: {length_of_lis(test)}")
    print(f"  贪心+二分: {length_of_lis_optimized(test)}")
    # 预期: [2, 5, 7, 101] → 4
