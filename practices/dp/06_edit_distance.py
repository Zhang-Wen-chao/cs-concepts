"""
编辑距离 - Edit Distance
LeetCode 72

问题：两个字符串 word1 → word2，最少编辑次数（插入/删除/替换）。

状态：dp[i][j] = word1[:i] → word2[:j] 的最小编辑距离
转移：
   如果 word1[i-1] == word2[j-1]：
      dp[i][j] = dp[i-1][j-1]
   否则：
      dp[i][j] = 1 + min(插入, 删除, 替换)
                        dp[i][j-1]  (word1 插入 word2[j-1])
                        dp[i-1][j]  (word1 删除 word1[i-1])
                        dp[i-1][j-1] (word1[i-1] 替换为 word2[j-1])
"""


# === 解法1：二维 DP O(m*n) / O(m*n) ===
def min_distance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 基础情况：空串
    for i in range(m + 1):
        dp[i][0] = i  # 删除所有字符
    for j in range(n + 1):
        dp[0][j] = j  # 插入所有字符

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],  # 插入
                    dp[i - 1][j],  # 删除
                    dp[i - 1][j - 1],  # 替换
                )

    return dp[m][n]


# === 解法2：空间压缩 O(m*n) / O(n) ===
def min_distance_optimized(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    if m < n:
        word1, word2 = word2, word1
        m, n = n, m

    dp = list(range(n + 1))  # dp[0][j] = j

    for i in range(1, m + 1):
        prev = dp[0]  # dp[i-1][0]
        dp[0] = i  # dp[i][0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if word1[i - 1] == word2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j - 1], dp[j], prev)
            prev = temp

    return dp[n]


if __name__ == "__main__":
    w1, w2 = "horse", "ros"
    print(f"word1 = '{w1}', word2 = '{w2}'")
    print(f"  标准 DP: {min_distance(w1, w2)}")
    print(f"  空间压缩: {min_distance_optimized(w1, w2)}")
    # 预期: 3 (horse→rorse→rose→ros)
