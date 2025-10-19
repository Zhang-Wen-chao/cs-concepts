# 近似算法 (Approximation Algorithms)

> 当完美太贵时，追求"足够好"

## 🎯 核心思想

近似算法解决一个现实问题：**NP完全问题无法快速求最优解，怎么办？**

**一句话理解：**
- 近似算法在多项式时间内找到"接近最优"的解
- 有质量保证：近似解 ≤ c × 最优解（c是近似比）

**为什么重要：**
```
很多实际问题是NP完全的：
• 物流配送（TSP）
• 任务调度
• 网络设计
• 资源分配

精确算法：
• 指数时间 → 大数据不可行
• n=100可能需要几千年

近似算法：
• 多项式时间 → 快速
• 有质量保证 → 可靠
• 实际效果常常很好
```

## 📖 近似比与性能保证

### 1. 什么是近似比？

#### 定义

```
最小化问题：
近似比 ρ(n) ≥ 1，满足：
  近似解 ≤ ρ(n) × 最优解

最大化问题：
近似比 ρ(n) ≥ 1，满足：
  近似解 ≥ (1/ρ(n)) × 最优解

ρ越接近1越好：
• ρ = 1：完美（精确算法）
• ρ = 2：2-近似
• ρ = 10：10-近似
```

#### 例子

```
TSP：
• 最优解：总距离100
• 2-近似算法：总距离≤200
• 实际可能：总距离150（更好）

顶点覆盖：
• 最优解：需要10个顶点
• 2-近似算法：使用≤20个顶点
• 实际可能：使用15个

关键：
• 保证的是最坏情况
• 实际常常更好
```

### 2. 近似模式 (Approximation Scheme)

#### PTAS（多项式时间近似模式）

```
对于任意ε > 0，算法在多项式时间内找到(1+ε)-近似解

时间复杂度：O(n^(1/ε))

例子：背包问题
• ε = 0.1 → 1.1-近似，时间O(n¹⁰)
• ε = 0.01 → 1.01-近似，时间O(n¹⁰⁰)

特点：
• 精度任意高
• 但时间随精度指数增长
```

#### FPTAS（完全多项式时间近似模式）

```
对于任意ε > 0，算法在O(poly(n, 1/ε))时间内找到(1+ε)-近似解

例子：背包问题的FPTAS
时间：O(n³/ε)

特点：
• 精度和规模都是多项式
• 最理想的近似算法类型
```

## 📖 经典近似算法

### 1. 顶点覆盖（2-近似）

#### 问题

```
输入：图G = (V, E)
目标：找最小顶点集S，使每条边至少有一个端点在S中

NP完全问题
```

#### 算法：贪心边选择

```python
def vertex_cover_approx(graph):
    """
    选择边，加入两个端点，删除相关边
    """
    cover = set()
    edges = set(graph.edges())

    while edges:
        # 选择任意边
        u, v = edges.pop()
        cover.add(u)
        cover.add(v)

        # 删除所有与u或v相关的边
        edges = {(a,b) for a,b in edges
                 if a not in (u,v) and b not in (u,v)}

    return cover

# 时间：O(E)
```

#### 分析

```
证明：近似比≤2

关键观察：
• 算法选择的边两两不相邻（无公共端点）
• 这些边在最优解中，每条至少需要覆盖一个端点
• 设算法选择k条边 → 算法解大小2k
• 最优解至少需要k个顶点（覆盖这k条不相邻的边）
• 近似比 = 2k/k = 2

实际：
• 理论保证2倍
• 实践中常常接近最优（1.2-1.5倍）
```

#### 为什么不能更好？

```
其他贪心策略：
1. 选择度数最大的顶点
   • 直观
   • 但近似比可以是O(log n)

2. 边匹配算法
   • 当前的算法
   • 保证2-近似

改进困难：
• 如果存在(2-ε)-近似多项式算法
  → P = NP（Khot猜想）
• 2-近似可能是最好的
```

### 2. 旅行商问题（TSP）

#### 问题

```
输入：完全图G，距离函数d
目标：找最短的访问所有顶点恰好一次的回路

NP-hard问题
```

#### 一般TSP：不可近似

```
定理：除非P=NP，一般TSP没有常数近似比算法

原因：
• TSP可以编码哈密尔顿回路问题
• 如果有近似算法，就能解决哈密尔顿回路
```

#### 度量TSP：2-近似

```python
def tsp_2_approximation(graph):
    """
    算法：
    1. 找最小生成树（MST）
    2. DFS遍历MST
    3. 按访问顺序连接
    """
    # 1. 最小生成树
    mst = minimum_spanning_tree(graph)

    # 2. DFS遍历
    def dfs(node, visited, tour):
        visited.add(node)
        tour.append(node)
        for neighbor in mst[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, tour)

    tour = []
    dfs(0, set(), tour)
    tour.append(0)  # 回到起点

    return tour

# 时间：O(V²)
```

#### 分析

```
三角不等式：d(u,w) ≤ d(u,v) + d(v,w)

证明：
• 设最优TSP距离为OPT
• MST距离 ≤ OPT（删除TSP的一条边 → 生成树）
• DFS遍历MST：每条边走两次 → 2×MST
• 走捷径（三角不等式）：≤ 2×MST ≤ 2×OPT

近似比：2

实际：
• 平均情况常常1.2-1.5倍最优
• 对实际数据效果好
```

#### Christofides算法：1.5-近似

```python
def tsp_christofides(graph):
    """
    改进版本：
    1. 找MST
    2. 找MST中奇度顶点
    3. 在奇度顶点间找最小完美匹配
    4. 合并MST和匹配，形成欧拉图
    5. 找欧拉回路，走捷径
    """
    # 1. MST
    mst = minimum_spanning_tree(graph)

    # 2. 奇度顶点
    odd_vertices = [v for v in graph.vertices()
                    if degree(v, mst) % 2 == 1]

    # 3. 最小完美匹配
    matching = min_perfect_matching(graph, odd_vertices)

    # 4. 合并
    multigraph = combine(mst, matching)

    # 5. 欧拉回路 + 走捷径
    euler_tour = find_euler_tour(multigraph)
    tour = shortcut(euler_tour)

    return tour

# 时间：O(V³)
# 近似比：1.5
```

#### 分析

```
关键观察：
• 奇度顶点个数一定是偶数
• 最小匹配 ≤ 0.5 × OPT

计算：
• MST ≤ OPT
• 匹配 ≤ 0.5 × OPT
• 总 = MST + 匹配 ≤ 1.5 × OPT

Christofides算法（1976）：
• 目前最好的多项式近似算法
• 40多年无人改进
• 是否存在<1.5的近似？开放问题
```

### 3. 集合覆盖（对数近似）

#### 问题

```
输入：全集U，子集族S = {S₁, S₂, ..., Sₘ}
目标：选择最少的子集，使其并集为U

例子：
U = {1,2,3,4,5}
S₁ = {1,2}, S₂ = {2,3,4}, S₃ = {4,5}, S₄ = {1,3,5}

最优解：{S₂, S₃}或{S₁, S₂, S₃}
```

#### 算法：贪心

```python
def set_cover_greedy(universe, subsets):
    """
    每次选择覆盖最多未覆盖元素的子集
    """
    covered = set()
    selected = []

    while covered != universe:
        # 选择覆盖最多新元素的子集
        best_set = max(subsets,
                       key=lambda s: len(s - covered))
        selected.append(best_set)
        covered |= best_set

    return selected

# 时间：O(|U| × |S|)
```

#### 分析

```
定理：贪心算法是H(n)-近似，其中H(n) = 1 + 1/2 + ... + 1/n

H(n) ≈ ln(n)

例子：
• n = 100 → H(100) ≈ 5
• n = 1000 → H(1000) ≈ 7

证明思路：
• 第i次选择，至少覆盖(未覆盖元素数/OPT)个元素
• 使用调和级数分析

下界：
• 除非P=NP，不存在(1-ε)ln(n)-近似
• 贪心算法接近最优
```

### 4. 背包问题（FPTAS）

#### 问题

```
输入：n个物品（价值v[i]，重量w[i]），容量W
目标：选择物品最大化价值，满足重量≤W

0-1背包：每个物品选或不选
```

#### 动态规划（伪多项式）

```python
def knapsack_dp(values, weights, W):
    """
    DP：dp[i][w] = 前i个物品，重量≤w的最大价值
    """
    n = len(values)
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(W + 1):
            # 不选第i个
            dp[i][w] = dp[i-1][w]
            # 选第i个
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i-1][w-weights[i-1]] + values[i-1])

    return dp[n][W]

# 时间：O(nW)
# 伪多项式：W可能很大
```

#### FPTAS：缩放价值

```python
def knapsack_fptas(values, weights, W, epsilon):
    """
    思想：缩放价值，使其变小
    """
    n = len(values)
    V_max = max(values)

    # 缩放因子
    K = epsilon * V_max / n

    # 缩放价值
    scaled_values = [int(v / K) for v in values]

    # 用DP解决缩放后的问题
    # 但这次DP[i][v] = 最小重量，价值≥v
    V_sum = sum(scaled_values)
    dp = [[float('inf')] * (V_sum + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for v in range(V_sum + 1):
            # 不选
            dp[i][v] = dp[i-1][v]
            # 选
            if v >= scaled_values[i-1]:
                dp[i][v] = min(dp[i][v],
                               dp[i-1][v-scaled_values[i-1]] + weights[i-1])

    # 找最大可行价值
    for v in range(V_sum, -1, -1):
        if dp[n][v] <= W:
            return v * K  # 恢复真实价值

# 时间：O(n³/ε)
# FPTAS！
```

#### 分析

```
近似保证：(1-ε)-近似

证明思路：
• 缩放损失：每个物品最多K
• n个物品：总损失≤nK = ε×V_max
• 近似解 ≥ OPT - nK ≥ OPT - ε×OPT = (1-ε)×OPT

精度与时间：
• ε = 0.1 → 90%最优，时间O(n³×10)
• ε = 0.01 → 99%最优，时间O(n³×100)

实际：
• 常见选择ε=0.05（95%精度）
• 对大多数应用足够好
```

### 5. 装箱问题（Bin Packing）

#### 问题

```
输入：n个物品（大小s[i]∈(0,1]）
目标：用最少的容量为1的箱子装下所有物品

例子：
物品：0.5, 0.6, 0.3, 0.4, 0.2
最优解：3个箱子
  箱1：0.5 + 0.4 = 0.9
  箱2：0.6 + 0.3 = 0.9
  箱3：0.2
```

#### First Fit算法

```python
def bin_packing_first_fit(items):
    """
    每个物品放入第一个能放下的箱子
    """
    bins = []

    for item in items:
        # 找第一个能放下的箱子
        placed = False
        for bin in bins:
            if sum(bin) + item <= 1:
                bin.append(item)
                placed = True
                break

        # 没有合适的箱子，开新箱
        if not placed:
            bins.append([item])

    return bins

# 时间：O(n²)
```

#### First Fit Decreasing（FFD）

```python
def bin_packing_ffd(items):
    """
    先排序（从大到小），再First Fit
    """
    sorted_items = sorted(items, reverse=True)
    return bin_packing_first_fit(sorted_items)

# 时间：O(n log n)
```

#### 分析

```
First Fit：
• 近似比：1.7 × OPT + 2
• 渐近：1.7-近似

First Fit Decreasing：
• 近似比：(11/9) × OPT + 6/9
• 渐近：1.222-近似
• 实际常常接近最优

下界：
• 没有PTAS（除非P=NP）
• 不存在<1.5的渐近近似（假设P≠NP）
```

## 🔧 设计近似算法的技巧

### 1. 贪心策略

```
思想：每步做局部最优选择

成功案例：
• 顶点覆盖：选边，加两端点
• 集合覆盖：选覆盖最多元素的集合
• 装箱：First Fit

注意：
• 不总是有好的近似比
• 需要仔细分析
```

### 2. 局部搜索

```python
def local_search(initial_solution, neighborhood):
    """
    从初始解开始，不断改进
    """
    current = initial_solution

    while True:
        # 找邻域中更好的解
        improved = False
        for neighbor in neighborhood(current):
            if cost(neighbor) < cost(current):
                current = neighbor
                improved = True
                break

        if not improved:
            break  # 局部最优

    return current

# k-opt局部搜索（TSP）
# 2-opt：交换两条边
# 3-opt：交换三条边
```

### 3. 线性规划舍入

```
思想：
1. 将问题表示为整数线性规划（ILP）
2. 松弛为线性规划（LP）
3. 求解LP（多项式时间）
4. 将LP解舍入为整数解

例子：顶点覆盖
ILP：
  minimize Σ xᵥ
  subject to xᵤ + xᵥ ≥ 1  (对每条边(u,v))
              xᵥ ∈ {0,1}

LP松弛：
  xᵥ ∈ [0,1]

舍入：
  如果xᵥ ≥ 0.5，选择v

分析：
• LP解 ≤ OPT
• 舍入解 ≤ 2 × LP解 ≤ 2 × OPT
```

### 4. 原始-对偶方法

```
思想：同时维护原始解和对偶解

应用：顶点覆盖、Steiner树、设施选址

优点：
• 统一的框架
• 常能得到好的近似比
```

## 💡 实践建议

### 1. 选择合适的算法

```
小数据（n < 30）：
• 精确算法（回溯、DP）
• 要求完美解

中等数据（30 < n < 10000）：
• FPTAS（如背包）
• 近似算法（如Christofides）

大数据（n > 10000）：
• 简单贪心（First Fit）
• 局部搜索（2-opt）

实时系统：
• 最快的近似算法
• 甚至启发式
```

### 2. 实际效果

```
理论近似比 vs 实际效果：

TSP 2-近似：
• 理论：最坏2倍
• 实际：平均1.2-1.3倍

顶点覆盖2-近似：
• 理论：最坏2倍
• 实际：平均1.3-1.5倍

原因：
• 理论分析最坏情况
• 实际数据有结构
• 平均情况更好
```

### 3. 混合策略

```python
def hybrid_approach(problem):
    """
    结合多种方法
    """
    # 1. 快速近似算法
    approx_solution = greedy(problem)

    # 2. 局部搜索改进
    local_solution = local_search(approx_solution)

    # 3. 如果时间允许，精确求解小子问题
    if problem.size < threshold:
        exact_solution = branch_and_bound(problem)
        return exact_solution

    return local_solution
```

## 🔗 与其他概念的联系

### 与NP完全性
- **NP完全** - 精确解难
- **近似算法** - 近似解容易

参考：`theory/complexity-theory/np-completeness.md`

### 与启发式算法
- **近似算法** - 有质量保证
- **启发式** - 无保证但实用

参考：`fundamentals/algorithms/`

### 与实际应用
- **物流** - TSP近似
- **调度** - 装箱近似
- **网络** - 顶点覆盖

## 📖 扩展阅读

### 高级主题
- 在线算法（Online Algorithms）
- 随机近似算法
- 次模优化
- 半定规划舍入

### 开放问题
- Unique Games Conjecture
- TSP能否<1.5近似？
- 各种问题的近似下界

---

**掌握近似算法，在完美与可行之间找到平衡！** 📊
