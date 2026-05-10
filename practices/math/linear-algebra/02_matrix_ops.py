"""
矩阵运算：乘法、转置、逆，验证 (AB)ᵀ = BᵀAᵀ
"""


def mat_mul(A, B):
    m, k = len(A), len(A[0])
    n = len(B[0])
    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]


def print_mat(name, M):
    print(f"{name}:")
    for row in M:
        print(f"  {row}")
    print()


if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[0, 1], [1, 0]]

    print_mat("A", A)
    print_mat("B", B)

    AB = mat_mul(A, B)
    print_mat("A×B", AB)

    # 验证 (AB)ᵀ = BᵀAᵀ
    AB_T = transpose(AB)
    BT_AT = mat_mul(transpose(B), transpose(A))
    print_mat("(AB)ᵀ", AB_T)
    print_mat("BᵀAᵀ", BT_AT)
    print(f"(AB)ᵀ == BᵀAᵀ: {AB_T == BT_AT}")
