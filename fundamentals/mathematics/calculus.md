# 微积分 - Calculus

> 研究变化率和累积量的数学，是优化和连续系统建模的基础

## 🎯 为什么要学微积分？

虽然计算机处理离散数据，但微积分在计算机科学中依然发挥重要作用：

- **机器学习**：梯度下降、反向传播、优化算法
- **计算机图形学**：曲线建模、物理仿真、光线追踪
- **算法分析**：连续化分析、积分估计、渐近分析
- **信号处理**：傅里叶变换、滤波器设计
- **数值计算**：微分方程求解、数值积分
- **经济建模**：成本函数优化、边际分析

**核心思想**：微分研究瞬时变化率，积分研究累积效应，它们互为逆运算。

## 📚 核心概念体系

### 1. [极限理论](#极限理论)
- 极限的概念和计算
- 连续性
- 中值定理

### 2. [导数](#导数)
- 导数的定义和几何意义
- 求导法则
- 高阶导数

### 3. [积分](#积分)
- 定积分和不定积分
- 积分的计算技巧
- 积分的应用

### 4. [优化理论](#优化理论)
- 函数的极值
- 拉格朗日乘数法
- 最优化算法

### 5. [微分方程](#微分方程)
- 常微分方程
- 偏微分方程基础
- 数值解法

---

## 极限理论

### 极限的概念

**极限**描述函数在某点附近的行为趋势：

$$\lim_{x \to a} f(x) = L$$

意思是：当x无限接近a时，f(x)无限接近L。

**ε-δ定义**：
对任意ε > 0，存在δ > 0，使得当0 < |x - a| < δ时，有|f(x) - L| < ε。

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_limit():
    """可视化极限概念"""
    # 函数 f(x) = (x² - 1)/(x - 1) 在 x = 1 处的极限
    x = np.linspace(0, 2, 1000)
    x = x[x != 1]  # 去除 x = 1
    y = (x**2 - 1) / (x - 1)  # 实际上等于 x + 1

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = (x²-1)/(x-1)')
    plt.axhline(y=2, color='red', linestyle='--', label='极限值 L = 2')
    plt.axvline(x=1, color='gray', linestyle=':', alpha=0.7)

    # 标记点 (1, 2) 为空心圆，表示函数在此处未定义
    plt.plot(1, 2, 'ro', markersize=8, fillstyle='none', markeredgewidth=2)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('极限概念：lim(x→1) (x²-1)/(x-1) = 2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.5, 1.5)
    plt.ylim(1.5, 2.5)
    plt.show()

visualize_limit()
```

### 重要极限

1. **基本极限**：
   - $\lim_{x \to 0} \frac{\sin x}{x} = 1$
   - $\lim_{x \to \infty} (1 + \frac{1}{x})^x = e$

2. **洛必达法则**：如果$\lim_{x \to a} f(x) = \lim_{x \to a} g(x) = 0$或$\infty$，则：
   $$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

```python
def important_limits():
    """验证重要极限"""
    # 验证 lim(x→0) sin(x)/x = 1
    x_values = np.array([0.1, 0.01, 0.001, 0.0001])
    sin_x_over_x = np.sin(x_values) / x_values

    print("验证 lim(x→0) sin(x)/x = 1:")
    for x, ratio in zip(x_values, sin_x_over_x):
        print(f"x = {x:6.4f}, sin(x)/x = {ratio:.6f}")

    print("\n验证 lim(n→∞) (1 + 1/n)^n = e:")
    # 验证 lim(n→∞) (1 + 1/n)^n = e
    n_values = np.array([10, 100, 1000, 10000, 100000])
    expressions = (1 + 1/n_values) ** n_values

    for n, expr in zip(n_values, expressions):
        print(f"n = {n:6d}, (1 + 1/n)^n = {expr:.6f}")
    print(f"e = {np.e:.6f}")

important_limits()
```

### 连续性

**连续函数**：在某点处极限值等于函数值
$$\lim_{x \to a} f(x) = f(a)$$

**连续性的重要定理**：
- **中间值定理**：连续函数在区间内取遍中间值
- **最值定理**：闭区间上的连续函数必有最大值和最小值

---

## 导数

### 导数的定义

**导数**表示函数在某点的瞬时变化率：

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**几何意义**：切线的斜率
**物理意义**：瞬时速度

### 基本求导法则

1. **基本函数导数**：
   - $(c)' = 0$ （常数）
   - $(x^n)' = nx^{n-1}$ （幂函数）
   - $(\sin x)' = \cos x$
   - $(\cos x)' = -\sin x$
   - $(e^x)' = e^x$
   - $(\ln x)' = \frac{1}{x}$

2. **运算法则**：
   - **加法**：$(f + g)' = f' + g'$
   - **乘法**：$(fg)' = f'g + fg'$
   - **除法**：$(\frac{f}{g})' = \frac{f'g - fg'}{g^2}$
   - **链式法则**：$(f(g(x)))' = f'(g(x)) \cdot g'(x)$

```python
import sympy as sp

def derivative_examples():
    """导数计算示例"""
    x = sp.Symbol('x')

    # 定义几个函数
    functions = [
        x**3 + 2*x**2 - 5*x + 1,
        sp.sin(x**2),
        sp.exp(x) * sp.cos(x),
        sp.ln(x**2 + 1)
    ]

    print("导数计算示例：")
    for i, f in enumerate(functions, 1):
        derivative = sp.diff(f, x)
        print(f"{i}. f(x) = {f}")
        print(f"   f'(x) = {derivative}")
        print()

derivative_examples()
```

### 导数的应用

**1. 梯度下降（机器学习核心算法）**

```python
def gradient_descent_demo():
    """梯度下降优化算法演示"""
    # 目标函数：f(x) = (x-3)² + 2
    # 导数：f'(x) = 2(x-3)

    def f(x):
        return (x - 3)**2 + 2

    def df_dx(x):
        return 2 * (x - 3)

    # 梯度下降
    x = 0.0  # 初始点
    learning_rate = 0.1
    history = [x]

    print("梯度下降过程：")
    print(f"初始点: x₀ = {x:.3f}, f(x₀) = {f(x):.3f}")

    for i in range(10):
        gradient = df_dx(x)
        x = x - learning_rate * gradient  # 更新规则
        history.append(x)
        print(f"步骤 {i+1}: x = {x:.3f}, f(x) = {f(x):.3f}, 梯度 = {gradient:.3f}")

    print(f"最优解附近: x ≈ {x:.3f}, f(x) ≈ {f(x):.3f}")

    # 可视化优化过程
    x_plot = np.linspace(-1, 6, 100)
    y_plot = f(x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = (x-3)² + 2')

    # 绘制优化路径
    history_y = [f(x) for x in history]
    plt.plot(history, history_y, 'ro-', markersize=6, label='梯度下降路径')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('梯度下降优化过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

gradient_descent_demo()
```

**2. 神经网络中的反向传播**

```python
def simple_backpropagation():
    """简单神经网络的反向传播演示"""
    # 简单的2层神经网络：x -> w1 -> sigmoid -> w2 -> y

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(z):
        return sigmoid(z) * (1 - sigmoid(z))

    # 训练数据
    x = 0.5
    y_true = 0.8

    # 初始权重
    w1 = 0.3
    w2 = 0.7

    print("神经网络反向传播演示：")
    print(f"输入: x = {x}, 目标输出: y = {y_true}")
    print(f"初始权重: w1 = {w1}, w2 = {w2}")

    learning_rate = 1.0

    for epoch in range(5):
        # 前向传播
        z1 = w1 * x
        a1 = sigmoid(z1)
        z2 = w2 * a1
        y_pred = sigmoid(z2)

        # 损失函数（均方误差）
        loss = 0.5 * (y_pred - y_true)**2

        # 反向传播
        # ∂L/∂w2 = (y_pred - y_true) * sigmoid'(z2) * a1
        dL_dw2 = (y_pred - y_true) * sigmoid_derivative(z2) * a1

        # ∂L/∂w1 = (y_pred - y_true) * sigmoid'(z2) * w2 * sigmoid'(z1) * x
        dL_dw1 = (y_pred - y_true) * sigmoid_derivative(z2) * w2 * sigmoid_derivative(z1) * x

        # 权重更新
        w2 -= learning_rate * dL_dw2
        w1 -= learning_rate * dL_dw1

        print(f"Epoch {epoch+1}: 预测={y_pred:.4f}, 损失={loss:.4f}, w1={w1:.4f}, w2={w2:.4f}")

simple_backpropagation()
```

---

## 积分

### 积分的概念

**不定积分**：导数的逆运算
$$\int f(x)dx = F(x) + C，其中F'(x) = f(x)$$

**定积分**：曲线下的面积
$$\int_a^b f(x)dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i)\Delta x$$

**牛顿-莱布尼兹公式**：连接定积分与不定积分
$$\int_a^b f(x)dx = F(b) - F(a)$$

### 基本积分公式

1. **基本函数积分**：
   - $\int x^n dx = \frac{x^{n+1}}{n+1} + C$ (n ≠ -1)
   - $\int \frac{1}{x} dx = \ln|x| + C$
   - $\int e^x dx = e^x + C$
   - $\int \sin x dx = -\cos x + C$
   - $\int \cos x dx = \sin x + C$

2. **积分技巧**：
   - **分部积分**：$\int u dv = uv - \int v du$
   - **换元法**：$\int f(g(x))g'(x)dx = \int f(u)du$，其中u = g(x)

```python
def numerical_integration():
    """数值积分方法演示"""
    # 计算 ∫₀¹ x² dx = 1/3

    def f(x):
        return x**2

    def riemann_sum(func, a, b, n):
        """黎曼和逼近定积分"""
        dx = (b - a) / n
        x_points = np.linspace(a + dx/2, b - dx/2, n)  # 中点法
        return dx * np.sum(func(x_points))

    def trapezoidal_rule(func, a, b, n):
        """梯形法则"""
        x = np.linspace(a, b, n+1)
        y = func(x)
        return (b - a) / (2 * n) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])

    def simpson_rule(func, a, b, n):
        """辛普森法则（n必须是偶数）"""
        if n % 2 != 0:
            n += 1
        x = np.linspace(a, b, n+1)
        y = func(x)
        return (b - a) / (3 * n) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

    true_value = 1/3
    n_values = [10, 100, 1000]

    print("数值积分方法比较 - 计算 ∫₀¹ x² dx = 1/3:")
    print(f"真实值: {true_value:.6f}")
    print()

    for n in n_values:
        riemann = riemann_sum(f, 0, 1, n)
        trapezoid = trapezoidal_rule(f, 0, 1, n)
        simpson = simpson_rule(f, 0, 1, n)

        print(f"n = {n}:")
        print(f"  黎曼和:   {riemann:.6f} (误差: {abs(riemann - true_value):.2e})")
        print(f"  梯形法则: {trapezoid:.6f} (误差: {abs(trapezoid - true_value):.2e})")
        print(f"  辛普森法则: {simpson:.6f} (误差: {abs(simpson - true_value):.2e})")
        print()

numerical_integration()
```

### 积分的应用

**1. 概率密度函数**
```python
def probability_applications():
    """积分在概率中的应用"""
    # 正态分布的累积分布函数
    from scipy.integrate import quad
    from scipy.stats import norm

    def normal_pdf(x, mu=0, sigma=1):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # 计算 P(-1 ≤ X ≤ 1) 其中 X ~ N(0,1)
    integral_result, _ = quad(normal_pdf, -1, 1)
    scipy_result = norm.cdf(1) - norm.cdf(-1)

    print("正态分布概率计算:")
    print(f"P(-1 ≤ X ≤ 1) = ∫₋₁¹ φ(x)dx")
    print(f"数值积分结果: {integral_result:.4f}")
    print(f"SciPy结果:    {scipy_result:.4f}")
    print(f"68-95-99.7法则预期: ≈0.68")

probability_applications()
```

---

## 优化理论

### 函数极值

**一阶条件**：$f'(x) = 0$ （必要条件）
**二阶条件**：
- $f''(x) > 0$：局部最小值
- $f''(x) < 0$：局部最大值
- $f''(x) = 0$：需要进一步分析

### 拉格朗日乘数法

用于求解约束优化问题：
$$\min f(x,y) \quad s.t. \quad g(x,y) = 0$$

**方法**：构造拉格朗日函数
$$L(x,y,\lambda) = f(x,y) - \lambda g(x,y)$$

**求解**：$\nabla L = 0$

```python
def lagrange_multiplier_example():
    """拉格朗日乘数法示例"""
    # 问题：在约束 x² + y² = 1 下，求 f(x,y) = x + y 的最值

    import sympy as sp

    x, y, lam = sp.symbols('x y lambda')

    # 目标函数和约束
    f = x + y
    g = x**2 + y**2 - 1

    # 拉格朗日函数
    L = f - lam * g

    # 求偏导数
    dL_dx = sp.diff(L, x)
    dL_dy = sp.diff(L, y)
    dL_dlam = sp.diff(L, lam)

    print("拉格朗日乘数法求解:")
    print(f"目标函数: f(x,y) = {f}")
    print(f"约束条件: g(x,y) = {g} = 0")
    print(f"拉格朗日函数: L = {L}")
    print()

    # 解方程组
    equations = [dL_dx, dL_dy, dL_dlam]
    solutions = sp.solve(equations, [x, y, lam])

    print("关键点:")
    for sol in solutions:
        x_val, y_val, lam_val = sol
        f_val = f.subs([(x, x_val), (y, y_val)])
        print(f"  ({x_val}, {y_val}): f = {f_val}")

    print("\n几何解释: 在单位圆上找直线 x + y = c 的切点")

lagrange_multiplier_example()
```

### 现代优化算法

```python
def modern_optimization():
    """现代优化算法演示：Adam算法"""
    # Adam算法：自适应学习率的梯度下降

    def rosenbrock(x):
        """Rosenbrock函数：经典测试函数"""
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def rosenbrock_gradient(x):
        """Rosenbrock函数的梯度"""
        dx1 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        dx2 = 200 * (x[1] - x[0]**2)
        return np.array([dx1, dx2])

    def adam_optimizer(grad_func, x_init, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        """Adam优化算法"""
        x = np.array(x_init, dtype=float)
        m = np.zeros_like(x)  # 一阶矩估计
        v = np.zeros_like(x)  # 二阶矩估计

        history = [x.copy()]

        for t in range(1, max_iter + 1):
            g = grad_func(x)  # 梯度

            # 更新偏差修正的一阶和二阶矩估计
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2

            # 偏差修正
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # 参数更新
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % 100 == 0:
                history.append(x.copy())

        return x, history

    # 优化Rosenbrock函数
    x_init = [-1.0, 1.0]
    print("Adam优化算法求解Rosenbrock函数:")
    print(f"初始点: {x_init}")
    print(f"理论最优解: (1, 1)")

    x_opt, history = adam_optimizer(rosenbrock_gradient, x_init)

    print(f"优化结果: ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"函数值: {rosenbrock(x_opt):.6f}")
    print(f"理论最优值: 0")

modern_optimization()
```

---

## 微分方程

### 常微分方程基础

**一阶线性微分方程**：
$$\frac{dy}{dx} + P(x)y = Q(x)$$

**解法**：积分因子法
$$y = e^{-\int P(x)dx} \left( \int Q(x)e^{\int P(x)dx}dx + C \right)$$

### 数值解法：欧拉方法

```python
def euler_method():
    """欧拉方法求解微分方程"""
    # 求解 dy/dx = -2y, y(0) = 1
    # 解析解: y = e^(-2x)

    def dydt(x, y):
        return -2 * y

    def analytical_solution(x):
        return np.exp(-2 * x)

    def euler_solve(func, x0, y0, h, x_end):
        """欧拉方法"""
        x_values = [x0]
        y_values = [y0]

        x, y = x0, y0
        while x < x_end:
            y = y + h * func(x, y)  # 欧拉公式
            x = x + h
            x_values.append(x)
            y_values.append(y)

        return np.array(x_values), np.array(y_values)

    # 不同步长的比较
    x_end = 2.0
    step_sizes = [0.1, 0.05, 0.01]

    print("欧拉方法求解 dy/dx = -2y, y(0) = 1:")
    print("解析解: y = e^(-2x)")
    print()

    plt.figure(figsize=(12, 8))

    # 解析解
    x_analytical = np.linspace(0, x_end, 200)
    y_analytical = analytical_solution(x_analytical)
    plt.plot(x_analytical, y_analytical, 'b-', linewidth=2, label='解析解')

    # 数值解
    for i, h in enumerate(step_sizes):
        x_num, y_num = euler_solve(dydt, 0, 1, h, x_end)
        plt.plot(x_num, y_num, 'o-', markersize=4, label=f'欧拉法 (h={h})')

        # 计算误差
        y_true_at_end = analytical_solution(x_end)
        error = abs(y_num[-1] - y_true_at_end)
        print(f"步长 h = {h}: 最终误差 = {error:.6f}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('欧拉方法求解微分方程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

euler_method()
```

---

## 🚀 实际应用案例

### 1. 神经网络训练中的优化

```python
def neural_network_training():
    """神经网络训练中的微积分应用"""
    # 简单的线性回归神经网络

    # 生成训练数据
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)  # y = 2x + 1 + noise

    # 网络参数
    w = np.random.randn()  # 权重
    b = np.random.randn()  # 偏置

    learning_rate = 0.01
    epochs = 100

    losses = []

    print("神经网络训练过程（梯度下降）:")
    print(f"真实参数: w = 2, b = 1")
    print(f"初始参数: w = {w:.3f}, b = {b:.3f}")

    for epoch in range(epochs):
        # 前向传播
        y_pred = w * X.flatten() + b

        # 损失函数（均方误差）
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)

        # 反向传播（计算梯度）
        dw = np.mean(2 * (y_pred - y) * X.flatten())
        db = np.mean(2 * (y_pred - y))

        # 参数更新
        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}, w = {w:.3f}, b = {b:.3f}")

    print(f"最终参数: w = {w:.3f}, b = {b:.3f}")

    # 绘制损失函数
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程中的损失函数')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.5, label='训练数据')
    x_plot = np.linspace(X.min(), X.max(), 100)
    y_plot = w * x_plot + b
    plt.plot(x_plot, y_plot, 'r-', label='学习到的直线')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

neural_network_training()
```

### 2. 计算机图形学中的曲线

```python
def parametric_curves():
    """参数曲线在计算机图形学中的应用"""

    def bezier_curve(P0, P1, P2, P3, t):
        """三次贝塞尔曲线"""
        return (1-t)**3 * P0 + 3*(1-t)**2*t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

    def bezier_derivative(P0, P1, P2, P3, t):
        """贝塞尔曲线的导数（切线方向）"""
        return -3*(1-t)**2 * P0 + 3*(1-t)**2 * P1 - 6*(1-t)*t * P1 + 6*(1-t)*t * P2 - 3*t**2 * P2 + 3*t**2 * P3

    # 控制点
    P0 = np.array([0, 0])
    P1 = np.array([1, 2])
    P2 = np.array([3, 2])
    P3 = np.array([4, 0])

    # 参数t
    t = np.linspace(0, 1, 100)

    # 计算曲线点
    curve_points = np.array([bezier_curve(P0, P1, P2, P3, ti) for ti in t])

    # 计算几个点的切线
    t_tangent = [0.2, 0.5, 0.8]
    tangent_points = []
    tangent_vectors = []

    for ti in t_tangent:
        point = bezier_curve(P0, P1, P2, P3, ti)
        tangent = bezier_derivative(P0, P1, P2, P3, ti)
        tangent = tangent / np.linalg.norm(tangent)  # 单位化

        tangent_points.append(point)
        tangent_vectors.append(tangent)

    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2, label='贝塞尔曲线')

    # 绘制控制点和控制多边形
    control_points = np.array([P0, P1, P2, P3])
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', alpha=0.5, label='控制点')

    # 绘制切线
    for i, (point, tangent) in enumerate(zip(tangent_points, tangent_vectors)):
        plt.arrow(point[0], point[1], 0.5*tangent[0], 0.5*tangent[1],
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        plt.plot(point[0], point[1], 'ro', markersize=8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('贝塞尔曲线与切线（导数的几何应用）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

    print("贝塞尔曲线在计算机图形学中的应用:")
    print("- 字体设计：字母轮廓的平滑曲线")
    print("- 动画：物体运动的轨迹")
    print("- CAD软件：工业设计中的曲面建模")
    print("- 游戏开发：角色移动的路径规划")

parametric_curves()
```

---

## 🚀 学习建议

1. **理解几何意义**：每个概念都要理解其几何和物理含义
2. **多练习计算**：熟练掌握基本的计算技巧
3. **重视应用**：将理论与计算机科学应用结合
4. **使用可视化**：用图形帮助理解抽象概念
5. **学会数值方法**：理论计算困难时用数值方法

## 🔗 相关概念

- [线性代数](linear-algebra.md) - 多元函数的微积分基础
- [概率统计](probability-statistics.md) - 连续分布和期望值计算
- [算法分析](../algorithms/) - 复杂度的渐近分析
- [机器学习](../../applications/artificial-intelligence/machine-learning/) - 优化算法的应用

---

**记住**：微积分不仅仅是数学工具，更是理解连续变化和优化问题的思维方式。在机器学习、计算机图形学等领域，微积分是不可或缺的基础！