"""
中心极限定理演示：均匀分布 → 正态分布
"""
import random
import math


def sample_mean(dist_func, n, num_samples=10000):
    """计算 n 个独立同分布样本均值的分布"""
    means = []
    for _ in range(num_samples):
        total = sum(dist_func() for _ in range(n))
        means.append(total / n)
    return means


def stats(samples):
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    return mean, math.sqrt(var)


if __name__ == "__main__":
    print("中心极限定理演示")
    print("从均匀分布 U(0,1) 采样，观察样本均值的分布\n")

    for n in [1, 2, 10, 30]:
        means = sample_mean(random.random, n, 5000)
        m, s = stats(means)
        # 理论：均值 = 0.5，标准差 = sqrt(1/12/n) ≈ 0.2887/sqrt(n)
        theo_mean = 0.5
        theo_std = math.sqrt(1 / 12 / n)
        print(f"n={n:2d}:  模拟均值={m:.4f}  理论均值={theo_mean:.4f}")
        print(f"         模拟标准差={s:.4f}  理论标准差={theo_std:.4f}")
        print(f"         形状: {'→ 趋近正态' if n >= 30 else '→ 还是均匀/三角'}")
        print()
