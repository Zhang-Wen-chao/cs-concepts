"""
复杂度估算练习 - Complexity Estimation

看代码片段，判断时间/空间复杂度。
"""


# === 练习1：O(1) ===
def is_empty(arr):
    """时间复杂度？"""
    return len(arr) == 0


# === 练习2：O(n) ===
def find_max(arr):
    """时间复杂度？"""
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val


# === 练习3：O(n²) ===
def bubble_sort(arr):
    """时间复杂度？"""
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


# === 练习4：O(log n) ===
def binary_search(arr, target):
    """时间复杂度？"""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# === 练习5：O(n log n) ===
def merge_sort(arr):
    """时间复杂度？空间复杂度？"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# === 练习6：O(2ⁿ) ===
def fibonacci(n):
    """时间复杂度？"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# === 练习7：O(m * n) ===
def print_all_pairs(arr1, arr2):
    """时间复杂度？(m = len(arr1), n = len(arr2))"""
    for i in arr1:
        for j in arr2:
            print(i, j)


if __name__ == "__main__":
    print("复杂度估算练习")
    print("=" * 40)
    print("is_empty:       O(1)")
    print("find_max:       O(n)")
    print("bubble_sort:    O(n²)")
    print("binary_search:  O(log n)")
    print("merge_sort:     O(n log n), space O(n)")
    print("fibonacci:      O(2ⁿ), space O(n) (栈)")
    print("print_all_pairs: O(m×n)")
