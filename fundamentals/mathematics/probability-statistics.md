# 概率与统计 - Probability & Statistics

> 研究不确定性和数据规律的数学，是机器学习和数据科学的核心基础

## 🎯 为什么要学概率统计？

在计算机科学中，我们经常面对不确定性和需要从数据中提取信息：

- **机器学习**：贝叶斯推理、概率图模型、参数估计
- **数据科学**：统计推断、假设检验、置信区间
- **算法分析**：随机算法、期望复杂度分析
- **网络安全**：密码学中的随机性、攻击概率评估
- **系统设计**：可靠性分析、性能建模
- **人工智能**：不确定性推理、决策理论

**核心思想**：概率帮我们量化不确定性，统计帮我们从有限样本推断总体规律。

## 📚 核心概念体系

### 1. [概率基础](#概率基础)
- 样本空间和事件
- 概率的定义和性质
- 条件概率和独立性

### 2. [随机变量](#随机变量)
- 离散随机变量
- 连续随机变量
- 期望值和方差

### 3. [重要分布](#重要分布)
- 离散分布：伯努利、二项、泊松
- 连续分布：均匀、正态、指数

### 4. [统计推断](#统计推断)
- 参数估计
- 假设检验
- 置信区间

### 5. [贝叶斯推理](#贝叶斯推理)
- 贝叶斯定理
- 先验和后验概率
- 贝叶斯网络

---

## 概率基础

### 什么是概率？

**概率**是对不确定事件发生可能性的数值度量，介于0和1之间。

**样本空间**($\Omega$)：所有可能结果的集合
**事件**(A)：样本空间的子集

**例子**：抛硬币
- 样本空间：$\Omega = \{正面, 反面\}$
- 事件A："得到正面" = $\{正面\}$
- $P(A) = 0.5$

### 概率的定义

**古典概率**（等可能情况）：
$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{有利结果数}}{\text{总结果数}}$$

**频率定义**（大数定律）：
$$P(A) = \lim_{n \to \infty} \frac{n(A)}{n}$$

**公理化定义**（Kolmogorov公理）：
1. $P(A) \geq 0$ 对所有事件A
2. $P(\Omega) = 1$
3. 对互斥事件：$P(A \cup B) = P(A) + P(B)$

```python
import numpy as np
import matplotlib.pyplot as plt

# 用频率逼近概率：抛硬币实验
def coin_flip_simulation(n_trials):
    """模拟抛硬币，观察频率如何逼近概率"""
    results = np.random.choice(['H', 'T'], n_trials)
    heads_count = np.cumsum(results == 'H')
    trials = np.arange(1, n_trials + 1)
    frequencies = heads_count / trials

    return trials, frequencies

# 运行模拟
trials, freqs = coin_flip_simulation(10000)
print(f"10000次试验后，正面频率: {freqs[-1]:.4f}")
print(f"理论概率: 0.5000")
```

### 条件概率

**定义**：在事件B发生的条件下，事件A发生的概率
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**乘法规则**：
$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

**全概率公式**：
$$P(A) = \sum_{i} P(A|B_i) \cdot P(B_i)$$

**例子**：疾病诊断
- $P(病) = 0.01$（患病率1%）
- $P(阳性|病) = 0.95$（敏感度95%）
- $P(阴性|健康) = 0.98$（特异度98%）

求：检测阳性时真的患病的概率？

```python
# 疾病诊断的贝叶斯计算
def medical_diagnosis():
    # 先验概率
    P_disease = 0.01
    P_healthy = 1 - P_disease

    # 似然概率
    P_positive_given_disease = 0.95    # 敏感度
    P_positive_given_healthy = 0.02    # 1 - 特异度

    # 全概率：P(阳性)
    P_positive = (P_positive_given_disease * P_disease +
                  P_positive_given_healthy * P_healthy)

    # 贝叶斯定理：P(病|阳性)
    P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

    print(f"检测阳性时患病概率: {P_disease_given_positive:.4f}")
    print(f"即使检测阳性，患病概率只有 {P_disease_given_positive:.1%}")

medical_diagnosis()
```

### 独立性

**事件独立**：A的发生不影响B发生的概率
$$P(A \cap B) = P(A) \cdot P(B)$$
$$P(A|B) = P(A)$$

**重要性质**：
- 独立不等于互斥（互斥事件通常是相关的）
- 独立性是对称的：A独立于B，则B也独立于A

---

## 随机变量

### 什么是随机变量？

**随机变量**是将样本空间中的结果映射到实数的函数。

**离散随机变量**：可能取值为有限个或可数无穷个
**连续随机变量**：可能取值为不可数无穷个（区间）

### 概率分布

**概率质量函数**（PMF，离散）：
$$P(X = x) = p(x)$$

**概率密度函数**（PDF，连续）：
$$P(a \leq X \leq b) = \int_a^b f(x)dx$$

**累积分布函数**（CDF）：
$$F(x) = P(X \leq x)$$

### 期望值和方差

**期望值**（均值）：随机变量的"平均"取值
$$E[X] = \begin{cases}
\sum_x x \cdot P(X = x) & \text{离散} \\
\int_{-\infty}^{\infty} x \cdot f(x)dx & \text{连续}
\end{cases}$$

**方差**：随机变量偏离均值的程度
$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**标准差**：$\sigma = \sqrt{\text{Var}(X)}$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 模拟离散随机变量：掷骰子
def dice_simulation(n_rolls=10000):
    """模拟掷骰子，计算期望和方差"""
    rolls = np.random.randint(1, 7, n_rolls)

    # 理论值
    theoretical_mean = (1 + 2 + 3 + 4 + 5 + 6) / 6  # 3.5
    theoretical_var = sum((x - 3.5)**2 for x in range(1, 7)) / 6  # 2.916

    # 实验值
    empirical_mean = np.mean(rolls)
    empirical_var = np.var(rolls)

    print(f"掷骰子 {n_rolls} 次:")
    print(f"理论期望: {theoretical_mean:.3f}, 实际: {empirical_mean:.3f}")
    print(f"理论方差: {theoretical_var:.3f}, 实际: {empirical_var:.3f}")

    return rolls

rolls = dice_simulation()
```

---

## 重要分布

### 离散分布

**1. 伯努利分布** Bernoulli(p)
- 单次试验，成功概率为p
- $P(X = 1) = p, P(X = 0) = 1-p$
- $E[X] = p, \text{Var}(X) = p(1-p)$

**2. 二项分布** Binomial(n, p)
- n次独立伯努利试验中成功的次数
- $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$
- $E[X] = np, \text{Var}(X) = np(1-p)$

**3. 泊松分布** Poisson(λ)
- 单位时间内随机事件发生次数
- $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$
- $E[X] = \lambda, \text{Var}(X) = \lambda$

```python
# 可视化重要的离散分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 伯努利分布
p = 0.3
x_bernoulli = [0, 1]
pmf_bernoulli = [1-p, p]
axes[0].bar(x_bernoulli, pmf_bernoulli)
axes[0].set_title(f'伯努利分布 (p={p})')
axes[0].set_xlabel('X')
axes[0].set_ylabel('P(X=k)')

# 二项分布
n, p = 10, 0.3
x_binomial = range(n+1)
pmf_binomial = [stats.binom.pmf(k, n, p) for k in x_binomial]
axes[1].bar(x_binomial, pmf_binomial)
axes[1].set_title(f'二项分布 (n={n}, p={p})')
axes[1].set_xlabel('X')

# 泊松分布
lam = 3
x_poisson = range(15)
pmf_poisson = [stats.poisson.pmf(k, lam) for k in x_poisson]
axes[2].bar(x_poisson, pmf_poisson)
axes[2].set_title(f'泊松分布 (λ={lam})')
axes[2].set_xlabel('X')

plt.tight_layout()
plt.show()
```

### 连续分布

**1. 均匀分布** Uniform(a, b)
- 在区间[a,b]上等概率分布
- $f(x) = \frac{1}{b-a}$ for $x \in [a,b]$
- $E[X] = \frac{a+b}{2}, \text{Var}(X) = \frac{(b-a)^2}{12}$

**2. 正态分布** Normal(μ, σ²)
- 最重要的连续分布，钟形曲线
- $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- $E[X] = \mu, \text{Var}(X) = \sigma^2$

**3. 指数分布** Exponential(λ)
- 描述事件间隔时间
- $f(x) = \lambda e^{-\lambda x}$ for $x \geq 0$
- $E[X] = \frac{1}{\lambda}, \text{Var}(X) = \frac{1}{\lambda^2}$

```python
# 正态分布的重要性质
def normal_distribution_properties():
    """演示正态分布的68-95-99.7法则"""
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, mu, sigma)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='N(0,1)')

    # 68-95-99.7法则
    for k, color in [(1, 'red'), (2, 'green'), (3, 'orange')]:
        x_fill = x[(x >= -k) & (x <= k)]
        y_fill = stats.norm.pdf(x_fill, mu, sigma)
        plt.fill_between(x_fill, y_fill, alpha=0.3, color=color,
                        label=f'±{k}σ: {stats.norm.cdf(k) - stats.norm.cdf(-k):.1%}')

    plt.xlabel('X')
    plt.ylabel('概率密度')
    plt.title('正态分布的68-95-99.7法则')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

normal_distribution_properties()
```

---

## 统计推断

### 参数估计

**目标**：从样本数据估计总体参数

**点估计**：用单个值估计参数
- **矩方法**：用样本矩估计总体矩
- **最大似然估计**(MLE)：使样本出现概率最大的参数值

**区间估计**：给出参数的可能范围
- **置信区间**：有一定把握包含真实参数的区间

```python
def parameter_estimation_example():
    """参数估计示例：估计正态分布的均值和方差"""
    # 生成真实数据（未知参数μ=5, σ=2）
    true_mu, true_sigma = 5, 2
    n_samples = 100
    data = np.random.normal(true_mu, true_sigma, n_samples)

    # 点估计
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # 无偏估计

    # 95%置信区间（均值）
    se = sample_std / np.sqrt(n_samples)  # 标准误差
    ci_lower = sample_mean - 1.96 * se
    ci_upper = sample_mean + 1.96 * se

    print(f"真实参数: μ={true_mu}, σ={true_sigma}")
    print(f"点估计: μ̂={sample_mean:.3f}, σ̂={sample_std:.3f}")
    print(f"均值95%置信区间: [{ci_lower:.3f}, {ci_upper:.3f}]")

    # 检查置信区间是否包含真实值
    contains_true = ci_lower <= true_mu <= ci_upper
    print(f"置信区间包含真实均值: {contains_true}")

parameter_estimation_example()
```

### 假设检验

**基本思想**：对总体参数提出假设，用样本数据检验假设是否成立

**步骤**：
1. 建立假设：$H_0$（原假设）vs $H_1$（备择假设）
2. 选择检验统计量
3. 确定显著性水平α（通常0.05）
4. 计算p值
5. 做出决策：拒绝或接受$H_0$

**两类错误**：
- **第I类错误**：拒绝正确的$H_0$（假阳性）
- **第II类错误**：接受错误的$H_0$（假阴性）

```python
def hypothesis_testing_example():
    """假设检验示例：检验硬币是否公平"""
    # 原假设：p = 0.5（硬币公平）
    # 备择假设：p ≠ 0.5（硬币不公平）

    n_flips = 100
    observed_heads = 60  # 观察到60次正面

    # 使用二项检验
    from scipy.stats import binom_test

    # 双边检验
    p_value = binom_test(observed_heads, n_flips, 0.5, alternative='two-sided')
    alpha = 0.05

    print(f"实验设置: 抛硬币{n_flips}次，观察到{observed_heads}次正面")
    print(f"原假设H0: p = 0.5 (硬币公平)")
    print(f"备择假设H1: p ≠ 0.5 (硬币不公平)")
    print(f"显著性水平α = {alpha}")
    print(f"p值 = {p_value:.4f}")

    if p_value < alpha:
        print("拒绝原假设，硬币可能不公平")
    else:
        print("无足够证据拒绝原假设，硬币可能是公平的")

hypothesis_testing_example()
```

---

## 贝叶斯推理

### 贝叶斯定理

贝叶斯定理描述如何根据新证据更新我们对假设的信念：

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

其中：
- $P(H|E)$：后验概率（看到证据后对假设的信念）
- $P(E|H)$：似然（假设为真时证据出现的概率）
- $P(H)$：先验概率（看到证据前对假设的信念）
- $P(E)$：证据的边际概率

### 贝叶斯推理过程

1. **先验**：基于已有知识对参数的初始信念
2. **似然**：给定参数值时观测数据的概率
3. **后验**：结合数据和先验后对参数的更新信念

**贝叶斯 vs 频率派**：
- **频率派**：参数是固定但未知的常数
- **贝叶斯派**：参数本身也是随机变量，有概率分布

```python
def bayesian_coin_flip():
    """贝叶斯硬币翻转：动态更新对硬币偏向性的信念"""
    # 先验：Beta(1,1) = 均匀分布，即对p没有偏见
    alpha_prior = 1
    beta_prior = 1

    # 观测数据：10次抛掷的结果
    observations = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]  # 1=正面, 0=反面

    alpha = alpha_prior
    beta = beta_prior

    print("贝叶斯硬币翻转推理过程：")
    print(f"先验分布: Beta({alpha_prior}, {beta_prior})")

    for i, obs in enumerate(observations, 1):
        # 更新后验分布参数
        if obs == 1:  # 观察到正面
            alpha += 1
        else:  # 观察到反面
            beta += 1

        # 计算后验均值和方差
        posterior_mean = alpha / (alpha + beta)
        posterior_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

        print(f"观测 {i}: {obs} -> 后验 Beta({alpha}, {beta}), "
              f"E[p] = {posterior_mean:.3f}, Var[p] = {posterior_var:.4f}")

bayesian_coin_flip()
```

### 贝叶斯网络

贝叶斯网络是表示变量间概率依赖关系的图模型：

```python
def simple_bayesian_network():
    """简单的医疗诊断贝叶斯网络"""
    # 网络结构：吸烟 -> 肺癌 -> X光异常
    #            肺癌 -> 咳嗽

    # 先验概率
    P_smoking = 0.3

    # 条件概率表
    # P(肺癌|吸烟)
    P_cancer_given_smoking = 0.1
    P_cancer_given_no_smoking = 0.01

    # P(咳嗽|肺癌)
    P_cough_given_cancer = 0.8
    P_cough_given_no_cancer = 0.2

    # P(X光异常|肺癌)
    P_xray_given_cancer = 0.9
    P_xray_given_no_cancer = 0.05

    # 推理：给定吸烟和咳嗽，求肺癌概率
    # P(肺癌|吸烟,咳嗽) ∝ P(咳嗽|肺癌) * P(肺癌|吸烟)

    # 计算联合概率
    # 情况1：吸烟，肺癌，咳嗽
    prob1 = P_smoking * P_cancer_given_smoking * P_cough_given_cancer

    # 情况2：吸烟，无肺癌，咳嗽
    prob2 = P_smoking * (1 - P_cancer_given_smoking) * P_cough_given_no_cancer

    # 归一化得到后验概率
    posterior_cancer = prob1 / (prob1 + prob2)

    print(f"吸烟且咳嗽的情况下，患肺癌的概率: {posterior_cancer:.3f}")

simple_bayesian_network()
```

---

## 🚀 实际应用案例

### 1. A/B测试

```python
def ab_test_analysis():
    """A/B测试的统计分析"""
    # 网站转化率测试
    # 版本A：1000用户，50转化
    # 版本B：1000用户，65转化

    n_a, x_a = 1000, 50  # A组样本量和转化数
    n_b, x_b = 1000, 65  # B组样本量和转化数

    p_a = x_a / n_a  # A组转化率
    p_b = x_b / n_b  # B组转化率

    # 计算差异的标准误差
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)

    # Z检验统计量
    z_score = (p_b - p_a) / se_diff

    # 双边检验的p值
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    print(f"A/B测试结果：")
    print(f"A组转化率: {p_a:.3f} ({x_a}/{n_a})")
    print(f"B组转化率: {p_b:.3f} ({x_b}/{n_b})")
    print(f"差异: {p_b - p_a:.3f}")
    print(f"Z统计量: {z_score:.3f}")
    print(f"p值: {p_value:.4f}")

    if p_value < 0.05:
        print("结论：B版本显著优于A版本")
    else:
        print("结论：没有显著差异")

ab_test_analysis()
```

### 2. 朴素贝叶斯分类器

```python
def naive_bayes_classifier():
    """朴素贝叶斯文本分类示例"""
    # 简化的垃圾邮件分类
    # 特征：邮件中包含特定词汇

    # 训练数据（词汇出现概率）
    # P(词汇|垃圾邮件)
    spam_word_probs = {
        'free': 0.8,
        'money': 0.7,
        'meeting': 0.1,
        'project': 0.05
    }

    # P(词汇|正常邮件)
    ham_word_probs = {
        'free': 0.2,
        'money': 0.1,
        'meeting': 0.6,
        'project': 0.7
    }

    # 先验概率
    P_spam = 0.4
    P_ham = 0.6

    # 测试邮件：包含 'free' 和 'project'
    test_words = ['free', 'project']

    # 计算似然（假设词汇独立）
    likelihood_spam = P_spam
    likelihood_ham = P_ham

    for word in test_words:
        likelihood_spam *= spam_word_probs.get(word, 0.01)  # 平滑
        likelihood_ham *= ham_word_probs.get(word, 0.01)

    # 归一化得到后验概率
    total_likelihood = likelihood_spam + likelihood_ham
    P_spam_given_words = likelihood_spam / total_likelihood
    P_ham_given_words = likelihood_ham / total_likelihood

    print(f"测试邮件包含词汇: {test_words}")
    print(f"P(垃圾邮件|词汇) = {P_spam_given_words:.3f}")
    print(f"P(正常邮件|词汇) = {P_ham_given_words:.3f}")

    if P_spam_given_words > P_ham_given_words:
        print("分类结果：垃圾邮件")
    else:
        print("分类结果：正常邮件")

naive_bayes_classifier()
```

---

## 🚀 学习建议

1. **数学基础要扎实**：概率论涉及大量积分和极限概念
2. **多做计算练习**：手算简单问题，培养直觉
3. **理解概念本质**：不要死记公式，要理解背后的逻辑
4. **结合实际应用**：每个概念都想想在CS中的用途
5. **使用编程验证**：用模拟实验验证理论结果

## 🔗 相关概念

- [线性代数](linear-algebra.md) - 多元统计的基础
- [离散数学](discrete-math.md) - 组合概率的基础
- [机器学习](../../applications/artificial-intelligence/machine-learning/) - 概率模型的应用
- [算法分析](../algorithms/) - 随机算法的分析

---

**记住**：概率统计不仅是数学工具，更是一种思维方式。它教会我们如何在不确定性中做出理性决策，这在数据驱动的现代计算机科学中至关重要！