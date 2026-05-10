"""
验证两个重要极限的数值逼近
"""
import math


def limit_sin_x_over_x():
    """lim(x->0) sin(x)/x = 1"""
    print("=== lim(x→0) sin(x)/x ===")
    for x in [0.1, 0.01, 0.001, 0.0001, 1e-6]:
        val = math.sin(x) / x
        print(f"  x = {x:.1e}  sin(x)/x = {val:.10f}")
    print()


def limit_one_plus_one_over_n_pow_n():
    """lim(n->inf) (1 + 1/n)^n = e"""
    print("=== lim(n→∞) (1 + 1/n)^n = e ===")
    for n in [1, 10, 100, 1000, 10000, 100000]:
        val = (1 + 1 / n) ** n
        print(f"  n = {n:6d}  (1+1/n)^n = {val:.10f}")
    print(f"  e ≈ {math.e:.10f}（真实值）")
    print()


if __name__ == "__main__":
    limit_sin_x_over_x()
    limit_one_plus_one_over_n_pow_n()
