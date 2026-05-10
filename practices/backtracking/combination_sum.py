"""
组合总和 - Combination Sum
LeetCode 39

问题：无重复数组 candidates，找所有和为 target 的组合（同一数字可用多次）。
"""


def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(start: int, path: list[int], remaining: int):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result


if __name__ == "__main__":
    cand = [2, 3, 6, 7]
    target = 7
    result = combination_sum(cand, target)
    print(f"combination_sum({cand}, {target}) = {result}")
