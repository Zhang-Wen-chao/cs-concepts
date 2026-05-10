"""
最长公共子序列 - Longest Common Subsequence (LCS)
LeetCode 1143

问题：两个字符串，找出最长公共子序列的长度。
不要求连续，保持相对顺序即可。

状态：dp[i][j] = text1[:i] 与 text2[:j] 的 LCS 长度
转移：
   如果 text1[i-1] == text2[j-1]：
      dp[i][j] = dp[i-1][j-1] + 1
   否则：
      dp[i][j] = max(dp[i-1][j], dp[i][j-1])
"""


# === 解法：二维 DP O(m*n) / O(m*n) ===
def longest_common_subsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# === 可选：空间压缩 O(m*n) / O(min(m,n)) ===
def longest_common_subsequence_optimized(text1: str, text2: str) -> int:
    # 保证 text2 是较短的，减少空间
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    m, n = len(text1), len(text2)

    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if text1[i - 1] == text2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


if __name__ == "__main__":
    s1, s2 = "abcde", "ace"
    print(f"text1 = '{s1}', text2 = '{s2}'")
    print(f"  标准 DP: {longest_common_subsequence(s1, s2)}")
    print(f"  空间压缩: {longest_common_subsequence_optimized(s1, s2)}")
    # 预期: "ace" → 3
