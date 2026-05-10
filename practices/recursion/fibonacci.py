"""
斐波那契数列 - Fibonacci

递归定义：F(n) = F(n-1) + F(n-2)
         F(0) = 0, F(1) = 1

展示了从暴力递归 → 记忆化 → DP 的进化。
"""


# 解法1：暴力递归 O(2^n)
def fib_recursive(n: int) -> int:
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


# 解法2：记忆化递归 O(n)
def fib_memo(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


# 解法3：迭代 DP O(n) / O(1)
def fib_iterative(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == "__main__":
    n = 10
    print(f"F({n}) =")
    print(f"  记忆化递归: {fib_memo(n)}")
    print(f"  迭代: {fib_iterative(n)}")

    # 小n可以跑暴力
    if n <= 30:
        print(f"  暴力递归: {fib_recursive(n)}")
