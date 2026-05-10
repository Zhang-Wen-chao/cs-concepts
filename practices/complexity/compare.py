"""
常见算法复杂度对比 - Algorithm Complexity Comparison

不同策略解决同一问题，复杂度天壤之别。
"""


# === 问题：斐波那契数列 ===

# 策略1：纯递归 O(2ⁿ) / O(n) 栈
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


# 策略2：记忆化递归 O(n) / O(n)
def fib_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


# 策略3：迭代 O(n) / O(1)
def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == "__main__":
    import time

    n = 35  # 递归版只能测小一点的数

    start = time.time()
    r1 = fib_recursive(n)
    t1 = time.time() - start

    start = time.time()
    r2 = fib_memo(n)
    t2 = time.time() - start

    start = time.time()
    r3 = fib_iterative(n)
    t3 = time.time() - start

    print(f"Fibonacci F({n})")
    print(f"  递归 O(2ⁿ):     {r1}  ({t1:.4f}s)")
    print(f"  记忆化 O(n):    {r2}  ({t2:.4f}s)")
    print(f"  迭代 O(n)/O(1): {r3}  ({t3:.4f}s)")
