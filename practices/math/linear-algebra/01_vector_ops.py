"""
向量运算：加减、点积、范数、夹角
"""
import math


def add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def subtract(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(v):
    return math.sqrt(dot(v, v))


def angle_cos(a, b):
    """cos(夹角) = a·b / (|a|·|b|)"""
    return dot(a, b) / (norm(a) * norm(b))


if __name__ == "__main__":
    v1 = [3, 4]
    v2 = [1, 2]

    print(f"向量 v1 = {v1}")
    print(f"向量 v2 = {v2}")
    print(f"v1 + v2 = {add(v1, v2)}")
    print(f"v1 - v2 = {subtract(v1, v2)}")
    print(f"v1 · v2 = {dot(v1, v2)}")
    print(f"|v1| = {norm(v1):.4f}")
    print(f"|v2| = {norm(v2):.4f}")
    print(f"cos(夹角) = {angle_cos(v1, v2):.4f}  (夹角 ≈ {math.degrees(math.acos(angle_cos(v1, v2))):.1f}°)")

    # 正交检验
    v3 = [1, 0]
    v4 = [0, 1]
    print(f"\n正交检验: v3={v3}, v4={v4}")
    print(f"  v3·v4 = {dot(v3, v4)}  (=0 → 正交)")
