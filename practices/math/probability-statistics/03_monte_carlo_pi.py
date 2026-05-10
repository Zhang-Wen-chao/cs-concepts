"""
蒙特卡洛模拟求 π
"""
import random
import math


def estimate_pi(n):
    inside = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside += 1
    return 4 * inside / n


if __name__ == "__main__":
    print("蒙特卡洛求 π")
    print(f"{'采样数':>8}  {'π 估计值':>12}  {'误差':>10}")
    print("-" * 35)

    for n in [100, 1000, 10000, 100000, 1000000]:
        pi_est = estimate_pi(n)
        error = abs(pi_est - math.pi)
        print(f"{n:8d}  {pi_est:12.6f}  {error:10.6f}")
    print(f"\n真实 π ≈ {math.pi:.6f}")
    print("规律：采样数越多 → 精度越高，但增长是 sqrt(n) 级")
