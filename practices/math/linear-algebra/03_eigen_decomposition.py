"""
幂法迭代求矩阵的主特征值和特征向量
"""
import math


def power_iteration(A, n_iters=100):
    """幂法：迭代求最大特征值对应的特征向量"""
    n = len(A)
    v = [1.0] * n  # 初始向量

    for _ in range(n_iters):
        # 矩阵乘向量
        w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        # 归一化
        norm = math.sqrt(sum(x * x for x in w))
        v = [x / norm for x in w]

    # 瑞利商求特征值
    Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
    lam = sum(Av[i] * v[i] for i in range(n))
    return lam, v


if __name__ == "__main__":
    # 对称矩阵，特征值应为 3 和 1
    A = [[2, 1],
         [1, 2]]

    lam, v = power_iteration(A)
    print(f"矩阵 A = [[2, 1], [1, 2]]")
    print(f"主特征值 λ ≈ {lam:.6f}  (理论值 3)")
    print(f"特征向量 v ≈ [{v[0]:.6f}, {v[1]:.6f}]  (理论 [1/√2, 1/√2])")
