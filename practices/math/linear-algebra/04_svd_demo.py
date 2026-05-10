"""
SVD 分解与低秩近似的误差分析
"""
import math


def svd_2x2(A):
    """2x2 矩阵的 SVD（手动简化版）"""
    # 计算 A^T A
    AT = [[A[0][0], A[1][0]], [A[0][1], A[1][1]]]
    ATA = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            ATA[i][j] = AT[i][0] * A[0][j] + AT[i][1] * A[1][j]

    # 特征值（简化平方根）
    trace = ATA[0][0] + ATA[1][1]
    det = ATA[0][0] * ATA[1][1] - ATA[0][1] * ATA[1][0]
    disc = trace * trace - 4 * det
    if disc < 0:
        disc = 0
    lam1 = (trace + math.sqrt(disc)) / 2
    lam2 = (trace - math.sqrt(disc)) / 2
    s1, s2 = math.sqrt(max(0, lam1)), math.sqrt(max(0, lam2))
    return [s1, s2]


if __name__ == "__main__":
    A = [[3, 0], [0, 1]]
    sigmas = svd_2x2(A)

    print("SVD 奇异值分析")
    print(f"矩阵 A = [[3, 0], [0, 1]]")
    print(f"奇异值: σ₁={sigmas[0]:.4f}, σ₂={sigmas[1]:.4f}")
    print()

    total = sum(s ** 2 for s in sigmas)
    for k in range(1, 3):
        energy = sum(sigmas[:k][i] ** 2 for i in range(k)) / total * 100
        print(f"用前 {k} 个奇异值: 信息保留 = {energy:.1f}%")
    print()
    print("结论：用 σ₁ 保留 90% 信息，可降至 1 维（图像压缩原理）")
