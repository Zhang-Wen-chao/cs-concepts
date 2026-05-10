"""
梯度下降求最小值：f(x) = (x-3)² + 2
"""
import math


def f(x):
    return (x - 3) ** 2 + 2


def grad(x):
    return 2 * (x - 3)


def gradient_descent(start, lr, steps):
    x = start
    print(f"初始 x₀ = {x:.4f}, f(x₀) = {f(x):.4f}")
    print(f"学习率 = {lr}\n")
    for i in range(steps):
        x = x - lr * grad(x)
        if i < 5 or i == steps - 1:
            print(f"  step {i+1:2d}: x = {x:.6f}, f(x) = {f(x):.6f}")
    print(f"\n最终 x ≈ {x:.6f}, f(x) ≈ {f(x):.6f} (理论最优 x=3, f=2)")
    return x


if __name__ == "__main__":
    gradient_descent(start=0.0, lr=0.1, steps=15)
    print()

    # 如果学习率太大，会发散
    print("=== 学习率过大导致发散 ===")
    gradient_descent(start=5.0, lr=0.9, steps=15)
