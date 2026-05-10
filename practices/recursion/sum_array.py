"""
数组求和 - Sum Array

递归：arr[0] + sum(arr[1:])
"""


def sum_array(arr: list) -> int:
    if not arr:
        return 0
    return arr[0] + sum_array(arr[1:])


if __name__ == "__main__":
    test = [1, 2, 3, 4, 5]
    print(f"sum({test}) = {sum_array(test)}")

    # 执行轨迹：
    # sum([1,2,3,4,5])
    # = 1 + sum([2,3,4,5])
    # = 1 + (2 + sum([3,4,5]))
    # = ... = 15
