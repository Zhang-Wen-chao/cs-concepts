"""
数值积分：矩形法 vs 梯形法，求 ∫₀¹ e^(-x²) dx
"""
import math


def f(x):
    return math.exp(-x * x)


def rectangle_method(a, b, n):
    """左矩形法"""
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        total += f(a + i * h)
    return total * h


def trapezoidal_method(a, b, n):
    """梯形法"""
    h = (b - a) / n
    total = (f(a) + f(b)) / 2
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


if __name__ == "__main__":
    a, b = 0.0, 1.0
    print(f"∫₀¹ e^(-x²) dx 数值积分对比\n")
    print(f"{'n':>6}  {'矩形法':>12}  {'梯形法':>12}  {'误差(梯形)':>12}")
    print("-" * 50)

    # 高精度梯形法作为参考值
    ref = trapezoidal_method(a, b, 1000000)

    for n in [10, 100, 1000, 10000]:
        rect = rectangle_method(a, b, n)
        trap = trapezoidal_method(a, b, n)
        err = abs(trap - ref)
        print(f"{n:6d}  {rect:12.8f}  {trap:12.8f}  {err:12.2e}")
