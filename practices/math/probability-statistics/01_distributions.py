"""
常见概率分布：期望和方差的理论值 vs 模拟值
"""
import random
import math


def bernoulli(p, n=100000):
    samples = [1 if random.random() < p else 0 for _ in range(n)]
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    return mean, var, f"伯努利(p={p})"


def binomial(n_trials, p, n=100000):
    samples = [sum(1 for _ in range(n_trials) if random.random() < p) for _ in range(n)]
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    return mean, var, f"二项(n={n_trials}, p={p})"


def uniform(a, b, n=100000):
    samples = [random.uniform(a, b) for _ in range(n)]
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    return mean, var, f"均匀(a={a}, b={b})"


def normal_dist(mu, sigma, n=100000):
    samples = [random.gauss(mu, sigma) for _ in range(n)]
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    return mean, var, f"正态(μ={mu}, σ={sigma})"


if __name__ == "__main__":
    print(f"{'分布':>25}  {'模拟均值':>10}  {'理论均值':>10}  {'模拟方差':>10}  {'理论方差':>10}")
    print("-" * 70)

    cases = [
        bernoulli(0.5),
        binomial(10, 0.5),
        uniform(0, 10),
        normal_dist(3, 2),
    ]

    for mean, var, name in cases:
        if "伯努利" in name:
            theo_mean = 0.5
            theo_var = 0.5 * 0.5
        elif "二项" in name:
            theo_mean = 10 * 0.5
            theo_var = 10 * 0.5 * 0.5
        elif "均匀" in name:
            theo_mean = (0 + 10) / 2
            theo_var = (10 - 0) ** 2 / 12
        else:
            theo_mean = 3
            theo_var = 4

        print(f"{name:>25}  {mean:10.4f}  {theo_mean:10.4f}  {var:10.4f}  {theo_var:10.4f}")
