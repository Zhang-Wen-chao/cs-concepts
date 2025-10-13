# Trees and Graphs - 树与图

> 层级关系和网络关系的数据结构

## 🌳 树 (Tree)

### 什么是树？

**有层次结构的数据结构，像一棵倒过来的树**

```
        根节点
         1
       /   \
      2     3      ← 子节点
     / \   / \
    4   5 6   7    ← 叶子节点
```

### 基本概念

- **根节点 (Root)**: 最顶端的节点
- **父节点 (Parent)**: 上一层的节点
- **子节点 (Child)**: 下一层的节点
- **叶子节点 (Leaf)**: 没有子节点的节点
- **深度 (Depth)**: 从根到节点的边数
- **高度 (Height)**: 从节点到叶子的最长路径
- **层级 (Level)**: 节点所在的层数

---

## 🌲 二叉树 (Binary Tree)

### 什么是二叉树？

**每个节点最多有两个子节点的树**

```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None   # 左子树
        self.right = None  # 右子树
```

### 二叉树的类型

#### 1. 满二叉树 (Full Binary Tree)

**每个节点要么有0个子节点，要么有2个子节点**

```
      1
    /   \
   2     3
  / \   / \
 4   5 6   7
```

#### 2. 完全二叉树 (Complete Binary Tree)

**除了最后一层，其他层都是满的，最后一层从左到右填充**

```
      1
    /   \
   2     3
  / \   /
 4   5 6
```

#### 3. 平衡二叉树 (Balanced Binary Tree)

**左右子树高度差不超过1**

```
      1
    /   \
   2     3
  /
 4

高度差：|height(left) - height(right)| ≤ 1
```

---

## 🔍 二叉树遍历

### 1. 前序遍历 (Pre-order: Root → Left → Right)

**根 → 左 → 右**

```python
def preorder(root):
    if not root:
        return []
    result = [root.val]
    result += preorder(root.left)
    result += preorder(root.right)
    return result

# 示例树:
#     1
#    / \
#   2   3
#  / \
# 4   5
# 结果: [1, 2, 4, 5, 3]
```

### 2. 中序遍历 (In-order: Left → Root → Right)

**左 → 根 → 右**

```python
def inorder(root):
    if not root:
        return []
    result = inorder(root.left)
    result.append(root.val)
    result += inorder(root.right)
    return result

# 结果: [4, 2, 5, 1, 3]
```

### 3. 后序遍历 (Post-order: Left → Right → Root)

**左 → 右 → 根**

```python
def postorder(root):
    if not root:
        return []
    result = postorder(root.left)
    result += postorder(root.right)
    result.append(root.val)
    return result

# 结果: [4, 5, 2, 3, 1]
```

### 4. 层序遍历 (Level-order: BFS)

**一层一层遍历**

```python
from collections import deque

def levelorder(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        result.append(node.val)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result

# 结果: [1, 2, 3, 4, 5]
```

### 遍历方式总结

```
树结构:
     1
    / \
   2   3
  / \
 4   5

前序: 1, 2, 4, 5, 3  (根左右)
中序: 4, 2, 5, 1, 3  (左根右)
后序: 4, 5, 2, 3, 1  (左右根)
层序: 1, 2, 3, 4, 5  (逐层)
```

---

## 🔎 二叉搜索树 (BST - Binary Search Tree)

### 什么是BST？

**左子树所有节点 < 根节点 < 右子树所有节点**

```
      5
    /   \
   3     7
  / \   / \
 2   4 6   8

特点：中序遍历是有序的
```

### BST操作

#### 查找

```python
def search(root, target):
    """O(log n)平均，O(n)最坏"""
    if not root or root.val == target:
        return root

    if target < root.val:
        return search(root.left, target)  # 去左边找
    else:
        return search(root.right, target)  # 去右边找
```

#### 插入

```python
def insert(root, val):
    """O(log n)平均"""
    if not root:
        return TreeNode(val)

    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)

    return root
```

#### 删除

```python
def delete(root, val):
    """O(log n)平均"""
    if not root:
        return None

    if val < root.val:
        root.left = delete(root.left, val)
    elif val > root.val:
        root.right = delete(root.right, val)
    else:
        # 找到要删除的节点
        # 情况1：叶子节点
        if not root.left and not root.right:
            return None
        # 情况2：只有一个子节点
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # 情况3：有两个子节点
        # 找右子树的最小值替换
        min_node = find_min(root.right)
        root.val = min_node.val
        root.right = delete(root.right, min_node.val)

    return root

def find_min(root):
    while root.left:
        root = root.left
    return root
```

---

## ⚖️ 平衡二叉树 (AVL Tree)

### 为什么需要平衡？

**不平衡的BST退化为链表，操作变成O(n)**

```
不平衡:          平衡:
  1              3
   \           /   \
    2         2     4
     \       /
      3     1
       \
        4
O(n)访问        O(log n)访问
```

### AVL树特点

- 左右子树高度差 ≤ 1
- 通过旋转保持平衡
- 所有操作保证O(log n)

### 旋转操作

```
左旋:
    y              x
   / \            / \
  x   C    →     A   y
 / \                / \
A   B              B   C

右旋:
    y              x
   / \            / \
  x   C    ←     A   y
 / \                / \
A   B              B   C
```

---

## 🔴 红黑树 (Red-Black Tree)

### 特点

- 每个节点是红色或黑色
- 根节点是黑色
- 所有叶子节点（NIL）是黑色
- 红色节点的子节点必须是黑色
- 从根到叶子的所有路径包含相同数量的黑色节点

**用途**：
- Java的TreeMap, TreeSet
- C++ STL的map, set
- Linux进程调度

---

## 🏔️ 堆 (Heap)

### 什么是堆？

**完全二叉树，满足堆性质**

### 最大堆 (Max Heap)

**父节点 ≥ 子节点**

```
      9
    /   \
   7     6
  / \   /
 3   5 4

根节点是最大值
```

### 最小堆 (Min Heap)

**父节点 ≤ 子节点**

```
      1
    /   \
   3     2
  / \   /
 7   5 4

根节点是最小值
```

### Python实现（最小堆）

```python
import heapq

# 创建堆
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
heapq.heappush(heap, 2)

# 堆顶元素（最小值）
print(heap[0])  # 1

# 弹出最小值
min_val = heapq.heappop(heap)  # 1

# 从列表创建堆
nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)  # O(n)
```

### 堆的应用

#### 1. 优先队列

```python
# 任务调度
tasks = [(2, "任务B"), (1, "任务A"), (3, "任务C")]
heapq.heapify(tasks)

while tasks:
    priority, task = heapq.heappop(tasks)
    print(f"执行 {task}")
# 输出: 任务A, 任务B, 任务C
```

#### 2. Top K问题

```python
def top_k_frequent(nums, k):
    """找出现频率最高的k个元素"""
    from collections import Counter
    count = Counter(nums)

    # 最小堆维护k个最大值
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)

    return [num for freq, num in heap]

print(top_k_frequent([1,1,1,2,2,3], 2))  # [1, 2]
```

#### 3. 合并K个有序链表

```python
def merge_k_sorted_lists(lists):
    heap = []
    # 初始化堆
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    dummy = TreeNode(0)
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

---

## 🔤 字典树 (Trie)

### 什么是Trie？

**用于存储字符串集合的树，共享公共前缀**

```
存储: ["cat", "car", "card", "dog"]

        root
       /    \
      c      d
      |      |
      a      o
     / \     |
    t   r    g
        |
        d
```

### Python实现

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """插入单词：O(m) m=单词长度"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        """查找完整单词：O(m)"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        """查找前缀：O(m)"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# 使用
trie = Trie()
trie.insert("apple")
trie.search("apple")      # True
trie.search("app")        # False
trie.starts_with("app")   # True
```

### 应用场景

- 自动补全
- 拼写检查
- IP路由
- 搜索引擎

---

## 🕸️ 图 (Graph)

### 什么是图？

**节点（顶点）和边的集合**

```
无向图:         有向图:
  A---B          A→B
  |\ /|          ↓ ↑
  | X |          C←D
  |/ \|
  C---D
```

### 图的类型

| 类型 | 特点 | 例子 |
|-----|------|------|
| **无向图** | 边没有方向 | 社交网络 |
| **有向图** | 边有方向 | 网页链接 |
| **加权图** | 边有权重 | 地图距离 |
| **无权图** | 边无权重 | 朋友关系 |

### 图的表示

#### 1. 邻接矩阵 (Adjacency Matrix)

```python
# 适合稠密图
graph = [
    [0, 1, 1, 0],  # A连接B,C
    [1, 0, 1, 1],  # B连接A,C,D
    [1, 1, 0, 1],  # C连接A,B,D
    [0, 1, 1, 0]   # D连接B,C
]

# 检查边：O(1)
has_edge = graph[0][1] == 1

# 空间：O(V²)
```

#### 2. 邻接表 (Adjacency List)

```python
# 适合稀疏图
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C']
}

# 检查边：O(V)
has_edge = 'B' in graph['A']

# 空间：O(V + E)
```

### 图的遍历

#### 1. 深度优先搜索 (DFS)

**尽可能深地搜索，用栈或递归**

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited

# 使用
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
dfs(graph, 'A')  # A B D E F C
```

#### 2. 广度优先搜索 (BFS)

**一层一层地搜索，用队列**

```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result

bfs(graph, 'A')  # ['A', 'B', 'C', 'D', 'E', 'F']
```

### DFS vs BFS对比

| 特性 | DFS | BFS |
|-----|-----|-----|
| **数据结构** | 栈/递归 | 队列 |
| **路径** | 不一定最短 | 最短路径 |
| **空间** | O(h) 高度 | O(w) 宽度 |
| **应用** | 拓扑排序、环检测 | 最短路径、层级遍历 |

---

## 🗺️ 图的经典算法

### 1. 拓扑排序

**有向无环图的线性排序**

```python
from collections import deque, defaultdict

def topological_sort(graph, n):
    """Kahn算法"""
    # 计算入度
    in_degree = defaultdict(int)
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # 入度为0的节点入队
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == n else []

# 课程安排问题
graph = {0: [1, 2], 1: [3], 2: [3], 3: []}
print(topological_sort(graph, 4))  # [0, 1, 2, 3] 或 [0, 2, 1, 3]
```

### 2. 环检测

```python
def has_cycle(graph):
    """DFS检测有向图中的环"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        if color[node] == GRAY:
            return True  # 找到环
        if color[node] == BLACK:
            return False  # 已访问

        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False
```

### 3. 最短路径 (Dijkstra)

```python
import heapq

def dijkstra(graph, start):
    """单源最短路径"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current_dist > distances[current]:
            continue

        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
```

---

## 📊 复杂度对比

| 数据结构 | 查找 | 插入 | 删除 | 空间 |
|---------|------|------|------|------|
| **BST(平均)** | O(log n) | O(log n) | O(log n) | O(n) |
| **BST(最坏)** | O(n) | O(n) | O(n) | O(n) |
| **AVL/红黑树** | O(log n) | O(log n) | O(log n) | O(n) |
| **堆** | O(n) | O(log n) | O(log n) | O(n) |
| **Trie** | O(m) | O(m) | O(m) | O(∑m) |

---

## 🔗 相关概念

- [栈与队列](stacks-queues.md) - DFS用栈，BFS用队列
- [哈希表](hash-tables.md) - 图的visited集合
- [算法基础](../algorithms/) - 图算法、树算法

---

**记住**：
1. 树 = 特殊的图（无环、连通）
2. BST中序遍历有序
3. 堆 = 优先队列
4. Trie = 字符串前缀树
5. DFS用栈，BFS用队列
6. 选择合适的图表示（邻接表 vs 邻接矩阵）
