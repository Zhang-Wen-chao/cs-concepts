"""
全排列 - Permutations
LeetCode 46

问题：给定不含重复数字的数组，返回所有排列。

回溯模板：做选择 → 递归 → 撤销选择
"""


def permute(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(path: list[int], remaining: list[int]):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i + 1:])
            path.pop()

    backtrack([], nums)
    return result


if __name__ == "__main__":
    nums = [1, 2, 3]
    result = permute(nums)
    print(f"permute({nums}) = {result}")
