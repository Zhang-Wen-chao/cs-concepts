"""
阶乘 - Factorial

递归定义：n! = n × (n-1)!
         0! = 1
"""


def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def factorial_iterative(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    n = 5
    print(f"{n}! = {factorial(n)}")
    print(f"{n}! (迭代) = {factorial_iterative(n)}")

    # 执行轨迹
    # factorial(5) = 5 * factorial(4)
    #            = 5 * (4 * factorial(3))
    #            = 5 * (4 * (3 * factorial(2)))
    #            = 5 * (4 * (3 * (2 * factorial(1))))
    #            = 5 * (4 * (3 * (2 * 1)))
    #            = 120
