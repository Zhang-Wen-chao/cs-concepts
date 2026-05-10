"""
幂运算 / 快速幂 - Power / Fast Power

递归版：base^n = base × base^(n-1)
快速幂：base^n = (base^(n/2))^2  → O(log n)
"""


def power(base: float, exp: int) -> float:
    """base^exp，O(n)"""
    if exp == 0:
        return 1
    return base * power(base, exp - 1)


def power_fast(base: float, exp: int) -> float:
    """快速幂，O(log n)"""
    if exp == 0:
        return 1
    half = power_fast(base, exp // 2)
    if exp % 2 == 0:
        return half * half
    else:
        return base * half * half


if __name__ == "__main__":
    for base, exp in [(2, 10), (3, 5), (5, 3)]:
        p1 = power(base, exp)
        p2 = power_fast(base, exp)
        print(f"{base}^{exp} = {p1}  (快速幂: {p2})")
        assert p1 == p2
