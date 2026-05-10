"""
时间 vs 空间权衡 - Time-Space Tradeoff

同一个问题，不同策略的复杂度对比。

问题：判断数组是否有重复元素。
"""


# 策略1：暴力 O(n²) / O(1)
def has_duplicate_v1(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False


# 策略2：排序 O(n log n) / O(1)
def has_duplicate_v2(nums):
    nums.sort()
    for i in range(len(nums) - 1):
        if nums[i] == nums[i + 1]:
            return True
    return False


# 策略3：哈希表 O(n) / O(n)
def has_duplicate_v3(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


if __name__ == "__main__":
    import time

    test = list(range(10000)) + [9999]  # 10k 个元素，有重复
    cases = [
        ("暴力 O(n²)", has_duplicate_v1),
        ("排序 O(n log n)", has_duplicate_v2),
        ("哈希 O(n)", has_duplicate_v3),
    ]

    for name, fn in cases:
        start = time.time()
        result = fn(test[:])  # 传拷贝避免排序影响
        elapsed = time.time() - start
        print(f"{name:20s}: {result}  ({elapsed:.4f}s)")
