"""
数值求导 vs 解析求导，验证链式法则
"""
import math


def numerical_derivative(f, x, h=1e-6):
    """中心差分法求数值导数"""
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x ** 3 + 2 * x ** 2 - 5 * x + 1


def f_prime(x):
    """解析导数 f'(x) = 3x² + 4x - 5"""
    return 3 * x * x + 4 * x - 5


# 链式法则验证：f(g(x)) 数值 vs 解析
def g(x):
    return x ** 2


def f_of_g(x):
    return f(g(x))


def f_of_g_prime_analytical(x):
    """解析链式：f'(g(x)) * g'(x)"""
    gx = g(x)
    return f_prime(gx) * (2 * x)


if __name__ == "__main__":
    print("=== 数值导数 vs 解析导数 ===")
    for x in [-2, 0, 1, 3]:
        num = numerical_derivative(f, x)
        ana = f_prime(x)
        print(f"  x={x:2d}  数值={num:.6f}  解析={ana:.6f}  误差={abs(num-ana):.2e}")

    print("\n=== 链式法则验证 ===")
    x = 2.0
    num = numerical_derivative(f_of_g, x)
    ana = f_of_g_prime_analytical(x)
    print(f"  f(g(x)) 在 x={x}: 数值={num:.6f}  解析={ana:.6f}  误差={abs(num-ana):.2e}")
