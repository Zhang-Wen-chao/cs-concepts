"""
子集 - Subsets
LeetCode 78

问题：给定不含重复数字的数组，返回所有子集。

回溯模板：每步决定选或不选当前元素。
"""


def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start: int, path: list[int]):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


if __name__ == "__main__":
    nums = [1, 2, 3]
    result = subsets(nums)
    print(f"subsets({nums}) = {result}")
