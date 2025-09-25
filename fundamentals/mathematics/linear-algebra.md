# 线性代数 - Linear Algebra

> 研究向量空间和线性变换的数学，是现代计算机科学的核心工具

## 🎯 为什么要学线性代数？

线性代数是计算机科学中应用最广泛的数学工具之一：

- **机器学习**：神经网络、主成分分析、支持向量机
- **计算机图形学**：3D变换、投影、旋转
- **数据科学**：降维、特征提取、数据压缩
- **计算机视觉**：图像处理、特征检测
- **搜索引擎**：PageRank算法
- **密码学**：RSA加密、椭圆曲线加密

**类比：** 如果说算术是处理数字，那么线性代数就是处理"数字的集合"（向量和矩阵）。它让我们能够优雅地处理高维数据。

## 📚 核心概念体系

### 1. [向量基础](#向量基础)
- 向量的概念和几何意义
- 向量运算：加法、数乘、点积
- 向量的长度和方向

### 2. [矩阵理论](#矩阵理论)
- 矩阵的基本概念
- 矩阵运算：加法、乘法、转置
- 特殊矩阵类型

### 3. [线性方程组](#线性方程组)
- 高斯消元法
- 矩阵的行变换
- 解的存在性和唯一性

### 4. [向量空间](#向量空间)
- 线性无关和线性相关
- 基和维数
- 子空间的概念

### 5. [线性变换](#线性变换)
- 变换的概念
- 特征值和特征向量
- 对角化

---

## 向量基础

### 什么是向量？

**向量**是有大小和方向的量，可以用来表示位移、速度、力等。

**几何角度**：向量是一个箭头
```
    ↗ v = (3, 2)
   /
  /
 /
O ———————→
```

**代数角度**：向量是一组有序数字
$$\vec{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

**编程角度**：向量是一个数组
```python
import numpy as np

# 2D向量
v = np.array([3, 2])

# 3D向量
u = np.array([1, -2, 4])
```

### 向量运算

**1. 向量加法**：对应分量相加
$$\vec{a} + \vec{b} = \begin{pmatrix} a_1 \\ a_2 \end{pmatrix} + \begin{pmatrix} b_1 \\ b_2 \end{pmatrix} = \begin{pmatrix} a_1 + b_1 \\ a_2 + b_2 \end{pmatrix}$$

**几何意义**：平行四边形法则
```python
a = np.array([2, 1])
b = np.array([1, 3])
result = a + b  # [3, 4]
```

**2. 数乘**：每个分量都乘以标量
$$c\vec{v} = c\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} cv_1 \\ cv_2 \end{pmatrix}$$

**几何意义**：改变向量的长度（和方向）
```python
v = np.array([2, 3])
scaled = 2 * v  # [4, 6] - 长度变为原来的2倍
```

**3. 点积（内积）**：度量向量的相似性
$$\vec{a} \cdot \vec{b} = a_1b_1 + a_2b_2 + \cdots + a_nb_n$$

**几何意义**：$\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}|\cos\theta$
- 点积 > 0：夹角为锐角
- 点积 = 0：垂直
- 点积 < 0：夹角为钝角

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
```

**应用例子**：
```python
# 计算向量长度（模）
def vector_length(v):
    return np.sqrt(np.dot(v, v))

# 计算两向量夹角
def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2) / (vector_length(v1) * vector_length(v2))
    return np.arccos(cos_angle)

# 检查两向量是否垂直
def is_perpendicular(v1, v2):
    return abs(np.dot(v1, v2)) < 1e-10
```

---

## 矩阵理论

### 什么是矩阵？

**矩阵**是按矩形阵列排列的数的集合，可以看作是向量的推广。

$$A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

**生活类比**：
- **电子表格**：每行是一个记录，每列是一个属性
- **图像**：每个元素是一个像素的亮度值
- **网络**：$A_{ij}$ 表示节点i到节点j是否有连接

```python
import numpy as np

# 创建矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2×3 矩阵

# 获取矩阵形状
print(A.shape)  # (2, 3)

# 访问元素
print(A[0, 1])  # 2 (第1行第2列，从0开始)
```

### 矩阵运算

**1. 矩阵加法**：对应元素相加（要求同型）
$$A + B = \begin{pmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{pmatrix}$$

**2. 数乘**：每个元素都乘以标量
$$cA = \begin{pmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{pmatrix}$$

**3. 矩阵乘法**：最重要的运算！

$(AB)_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$

**记忆口诀**：第i行乘第j列

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# 矩阵乘法
C = np.matmul(A, B)  # 或者 A @ B
print(C)
# [[19 22]   # 1*5+2*7=19, 1*6+2*8=22
#  [43 50]]  # 3*5+4*7=43, 3*6+4*8=50
```

**矩阵乘法的几何意义**：线性变换的复合

**4. 转置**：行列互换
$$A^T: (A^T)_{ij} = A_{ji}$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T  # 或 np.transpose(A)
print(A_T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### 特殊矩阵

**1. 单位矩阵**：矩阵乘法的"1"
$$I = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

性质：$AI = IA = A$

**2. 零矩阵**：矩阵加法的"0"

**3. 逆矩阵**：满足 $AA^{-1} = I$ 的矩阵 $A^{-1}$

```python
# 创建单位矩阵
I = np.eye(3)

# 计算逆矩阵（如果存在）
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

# 验证
print(np.allclose(A @ A_inv, np.eye(2)))  # True
```

---

## 线性方程组

线性代数最初就是为了解线性方程组而发展的！

### 从方程组到矩阵

考虑线性方程组：
$$\begin{cases}
2x + 3y = 7 \\
x - y = 1
\end{cases}$$

可以写成矩阵形式：
$$\begin{pmatrix} 2 & 3 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 7 \\ 1 \end{pmatrix}$$

即：$A\vec{x} = \vec{b}$

### 高斯消元法

**思想**：通过行变换将矩阵化为阶梯形

```python
def gauss_elimination(A, b):
    """
    高斯消元法求解线性方程组 Ax = b
    """
    n = len(b)
    # 增广矩阵
    augmented = np.column_stack([A, b])

    # 前向消元
    for i in range(n):
        # 选择主元（避免数值不稳定）
        max_row = np.argmax(np.abs(augmented[i:, i])) + i
        augmented[[i, max_row]] = augmented[[max_row, i]]

        # 消元
        for j in range(i+1, n):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j] -= factor * augmented[i]

    # 回代求解
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (augmented[i, -1] - np.dot(augmented[i, i+1:n], x[i+1:n])) / augmented[i, i]

    return x

# 示例
A = np.array([[2, 3], [1, -1]], dtype=float)
b = np.array([7, 1], dtype=float)
solution = gauss_elimination(A, b)
print(f"解: x = {solution}")  # [2, 1]
```

### 解的情况

对于 $A\vec{x} = \vec{b}$：

1. **唯一解**：$A$ 可逆（行列式非零）
2. **无解**：矛盾方程组
3. **无穷多解**：欠定方程组

```python
# 使用numpy求解
A = np.array([[2, 3], [1, -1]])
b = np.array([7, 1])

# 方法1：逆矩阵法（仅当A可逆时）
if np.linalg.det(A) != 0:
    x = np.linalg.inv(A) @ b

# 方法2：numpy的线性求解器（推荐）
x = np.linalg.solve(A, b)
print(f"解: {x}")  # [2. 1.]
```

---

## 向量空间

### 线性组合与线性相关

**线性组合**：
$$c_1\vec{v_1} + c_2\vec{v_2} + \cdots + c_n\vec{v_n}$$

**线性无关**：只有当所有系数都为0时，线性组合才等于零向量

**几何理解**：
- 2D中：两个非共线向量线性无关
- 3D中：三个非共面向量线性无关

```python
def is_linearly_independent(vectors):
    """
    检查向量组是否线性无关
    方法：将向量作为列组成矩阵，计算其秩
    """
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    return rank == len(vectors)

# 示例
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
print(is_linearly_independent([v1, v2, v3]))  # True

v4 = np.array([1, 1, 0])  # v4 = v1 + v2，线性相关
print(is_linearly_independent([v1, v2, v4]))  # False
```

### 基与维数

**基**：向量空间中线性无关的向量组，能张成整个空间

**标准基**（3D空间）：
$$\vec{e_1} = \begin{pmatrix}1\\0\\0\end{pmatrix}, \vec{e_2} = \begin{pmatrix}0\\1\\0\end{pmatrix}, \vec{e_3} = \begin{pmatrix}0\\0\\1\end{pmatrix}$$

**维数**：基中向量的个数

```python
# 标准基
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# 任意向量都可以用基表示
v = np.array([3, -2, 5])
# v = 3*e1 + (-2)*e2 + 5*e3

# 基的变换
def change_of_basis(vector, old_basis, new_basis):
    """
    将向量从一组基表示转换到另一组基
    """
    old_matrix = np.column_stack(old_basis)
    new_matrix = np.column_stack(new_basis)

    # 先转换为标准坐标，再转换为新基坐标
    standard_coords = old_matrix @ vector
    new_coords = np.linalg.inv(new_matrix) @ standard_coords
    return new_coords
```

---

## 线性变换

### 什么是线性变换？

**线性变换**是保持向量加法和数乘的函数：
- $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$
- $T(c\vec{v}) = cT(\vec{v})$

**每个线性变换都可以用矩阵表示**！

### 常见的2D变换

```python
import matplotlib.pyplot as plt

# 原始向量
v = np.array([2, 1])

# 1. 旋转矩阵（逆时针旋转θ度）
def rotation_matrix(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])

# 旋转45度
theta = np.pi / 4
R = rotation_matrix(theta)
rotated_v = R @ v

# 2. 缩放矩阵
S = np.array([[2, 0],    # x方向放大2倍
              [0, 0.5]])  # y方向缩小到0.5倍
scaled_v = S @ v

# 3. 反射矩阵（关于x轴）
Reflect_x = np.array([[1, 0],
                      [0, -1]])
reflected_v = Reflect_x @ v

# 4. 切变（剪切）
Shear = np.array([[1, 1],  # y方向不变，x方向加上y的分量
                  [0, 1]])
sheared_v = Shear @ v

print(f"原向量: {v}")
print(f"旋转后: {rotated_v}")
print(f"缩放后: {scaled_v}")
print(f"反射后: {reflected_v}")
print(f"切变后: {sheared_v}")
```

### 特征值和特征向量

**定义**：对于矩阵 $A$，如果存在非零向量 $\vec{v}$ 和标量 $\lambda$ 使得：
$$A\vec{v} = \lambda\vec{v}$$

那么 $\lambda$ 是特征值，$\vec{v}$ 是对应的特征向量。

**几何意义**：特征向量的方向在变换后不变，只是长度被缩放了 $\lambda$ 倍。

```python
# 计算特征值和特征向量
A = np.array([[3, 1],
              [0, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]

    print(f"验证 λ={lambda_i:.2f}:")
    print(f"A*v = {A @ v_i}")
    print(f"λ*v = {lambda_i * v_i}")
    print(f"误差: {np.allclose(A @ v_i, lambda_i * v_i)}")
    print()
```

### 应用：主成分分析(PCA)

PCA是数据降维的经典方法，基于特征值分解：

```python
def pca_example():
    """
    简单的PCA示例
    """
    # 生成2D数据
    np.random.seed(42)
    data = np.random.multivariate_normal([0, 0], [[3, 1], [1, 1]], 100)

    # 1. 中心化数据
    data_centered = data - np.mean(data, axis=0)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(data_centered.T)

    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 按特征值大小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 选择主成分（这里选择第一个）
    principal_component = eigenvectors[:, 0]

    # 6. 投影到主成分方向
    projected_data = data_centered @ principal_component.reshape(-1, 1)

    print(f"原始数据维度: {data.shape}")
    print(f"降维后数据维度: {projected_data.shape}")
    print(f"主成分方向: {principal_component}")
    print(f"解释的方差比例: {eigenvalues[0] / np.sum(eigenvalues):.2%}")

pca_example()
```

---

## 🚀 实际应用案例

### 1. 图像处理：图像旋转

```python
def rotate_image_matrix():
    """
    使用矩阵乘法旋转图像
    """
    # 创建简单的3x3图像（用数值表示）
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # 90度顺时针旋转矩阵
    # 对于图像，我们需要考虑坐标系统
    rotation_90_cw = np.array([[0, 1],
                               [-1, 0]])

    # 实际中图像旋转涉及坐标变换和插值
    # 这里展示概念
    print("原图像:")
    print(image)
    print("\n旋转90度后:")
    print(np.rot90(image, -1))  # numpy的旋转函数

rotate_image_matrix()
```

### 2. 推荐系统：矩阵分解

```python
def simple_matrix_factorization():
    """
    简化的矩阵分解用于推荐系统
    """
    # 用户-物品评分矩阵（0表示未评分）
    ratings = np.array([[5, 3, 0, 1],
                        [4, 0, 0, 1],
                        [1, 1, 0, 5],
                        [1, 0, 0, 4],
                        [0, 1, 5, 4]])

    print("用户-物品评分矩阵:")
    print(ratings)

    # 实际的矩阵分解算法（如SVD）可以用来
    # 发现潜在特征并预测缺失评分
    U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

    # 重构矩阵（降维后）
    k = 2  # 保留前2个奇异值
    reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    print(f"\n重构后的矩阵（保留{k}个主要特征）:")
    print(reconstructed)

simple_matrix_factorization()
```

---

## 🚀 学习建议

1. **先理解几何直觉**：向量和矩阵的几何意义比代数运算更重要
2. **多练习计算**：手算小规模问题，培养数感
3. **学会用工具**：NumPy是线性代数的利器
4. **连接应用**：每学一个概念就想想它在CS中的应用
5. **可视化理解**：用图形帮助理解变换和空间概念

## 🔗 相关概念

- [离散数学](discrete-math.md) - 数学基础的另一个重要分支
- [概率统计](probability-statistics.md) - 多元统计需要线性代数
- [机器学习](../../applications/artificial-intelligence/machine-learning/) - 大量使用线性代数
- [计算机视觉](../../applications/artificial-intelligence/computer-vision/) - 图像变换基于线性代数

---

**记住**：线性代数不只是计算，更是一种思维方式。它让我们能够优雅地处理高维数据，理解空间结构，这在现代计算机科学中无处不在！