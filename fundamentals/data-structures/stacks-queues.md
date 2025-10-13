# Stacks and Queues - 栈与队列

> 限制访问方式的线性结构：后进先出 vs 先进先出

## 🎯 核心区别

| 特性 | 栈 (Stack) | 队列 (Queue) |
|-----|-----------|-------------|
| **原则** | 后进先出 (LIFO) | 先进先出 (FIFO) |
| **比喻** | 叠盘子 | 排队 |
| **插入** | push (压栈) | enqueue (入队) |
| **删除** | pop (弹栈) | dequeue (出队) |

---

## 📚 栈 (Stack)

### 什么是栈？

**后进先出 (Last In First Out, LIFO)** 的数据结构

```
栈的操作：

Push (入栈):          Pop (出栈):
     [4]                  [4] ← 弹出
     [3]                  [3]
     [2]                  [2]
     [1]                  [1]
    ─────                ─────
```

### 生活中的栈

- 📚 一摞书：拿走最上面的
- 🍽️ 叠盘子：取最上面的
- ⌨️ Ctrl+Z：撤销最近的操作
- 🌐 浏览器后退：回到上一个页面

### 基本操作

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        """入栈：O(1)"""
        self.items.append(item)

    def pop(self):
        """出栈：O(1)"""
        if self.is_empty():
            raise IndexError("栈为空")
        return self.items.pop()

    def peek(self):
        """查看栈顶元素：O(1)"""
        if self.is_empty():
            raise IndexError("栈为空")
        return self.items[-1]

    def is_empty(self):
        """判断是否为空：O(1)"""
        return len(self.items) == 0

    def size(self):
        """获取大小：O(1)"""
        return len(self.items)

# 使用
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)

print(stack.peek())  # 3 (最后进入的)
print(stack.pop())   # 3
print(stack.pop())   # 2
print(stack.pop())   # 1
```

### 栈的实现方式

#### 1. 基于数组

```python
# 用Python list实现（最常见）
class ArrayStack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)  # O(1) 摊销

    def pop(self):
        return self.items.pop()  # O(1)
```

#### 2. 基于链表

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedStack:
    def __init__(self):
        self.top = None
        self.size = 0

    def push(self, item):
        """在头部插入：O(1)"""
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self.size += 1

    def pop(self):
        """从头部删除：O(1)"""
        if not self.top:
            raise IndexError("栈为空")
        data = self.top.data
        self.top = self.top.next
        self.size -= 1
        return data
```

### 栈的经典应用

#### 1. 括号匹配

```python
def is_balanced(expression):
    """检查括号是否匹配"""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for char in expression:
        if char in '([{':
            stack.append(char)  # 左括号入栈
        elif char in ')]}':
            if not stack or stack.pop() != pairs[char]:
                return False  # 不匹配
    return len(stack) == 0  # 栈应该为空

# 测试
print(is_balanced("()"))        # True
print(is_balanced("()[]{}"))    # True
print(is_balanced("(]"))        # False
print(is_balanced("([)]"))      # False
print(is_balanced("{[()]}"))    # True
```

#### 2. 函数调用栈

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# 调用 factorial(3) 的栈过程：
"""
调用栈：
factorial(3)
  factorial(2)
    factorial(1)
      factorial(0) → 返回1
    ← 返回 1 * 1 = 1
  ← 返回 2 * 1 = 2
← 返回 3 * 2 = 6
"""
```

#### 3. 表达式求值

```python
def evaluate_postfix(expression):
    """计算后缀表达式（逆波兰表示法）"""
    stack = []

    for token in expression.split():
        if token.isdigit():
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a // b)

    return stack.pop()

# 测试
# "3 4 + 2 *" 相当于 (3 + 4) * 2 = 14
print(evaluate_postfix("3 4 + 2 *"))  # 14
```

#### 4. 浏览器历史记录

```python
class BrowserHistory:
    def __init__(self):
        self.back_stack = []
        self.forward_stack = []
        self.current = None

    def visit(self, url):
        """访问新页面"""
        if self.current:
            self.back_stack.append(self.current)
        self.current = url
        self.forward_stack = []  # 清空前进栈

    def back(self):
        """后退"""
        if not self.back_stack:
            return None
        self.forward_stack.append(self.current)
        self.current = self.back_stack.pop()
        return self.current

    def forward(self):
        """前进"""
        if not self.forward_stack:
            return None
        self.back_stack.append(self.current)
        self.current = self.forward_stack.pop()
        return self.current
```

---

## 🚶 队列 (Queue)

### 什么是队列？

**先进先出 (First In First Out, FIFO)** 的数据结构

```
队列的操作：

Enqueue (入队):               Dequeue (出队):
Front → [1][2][3][4] ← Rear   Front → [2][3][4] ← Rear
              ↑ 新元素加这里           ↑ 从这里移除
```

### 生活中的队列

- 🎫 排队买票：先来先服务
- 🖨️ 打印队列：先提交先打印
- 📞 客服热线：按顺序接听
- 🚦 车辆排队：先到先过

### 基本操作

```python
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        """入队（在尾部添加）：O(1)"""
        self.items.append(item)

    def dequeue(self):
        """出队（从头部移除）：O(n) - 慢！"""
        if self.is_empty():
            raise IndexError("队列为空")
        return self.items.pop(0)  # 移除第一个元素，需要移动所有元素

    def front(self):
        """查看队首元素：O(1)"""
        if self.is_empty():
            raise IndexError("队列为空")
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
```

### 高效的队列实现

#### 使用collections.deque

```python
from collections import deque

class EfficientQueue:
    def __init__(self):
        self.items = deque()  # 双端队列

    def enqueue(self, item):
        self.items.append(item)  # O(1)

    def dequeue(self):
        if not self.items:
            raise IndexError("队列为空")
        return self.items.popleft()  # O(1) - 快！

# Python标准库推荐的队列实现
queue = deque()
queue.append(1)      # 入队
queue.append(2)
value = queue.popleft()  # 出队
```

#### 基于链表

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedQueue:
    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0

    def enqueue(self, item):
        """在尾部添加：O(1)"""
        new_node = Node(item)
        if self.rear:
            self.rear.next = new_node
        self.rear = new_node
        if not self.front:
            self.front = new_node
        self.size += 1

    def dequeue(self):
        """从头部移除：O(1)"""
        if not self.front:
            raise IndexError("队列为空")
        data = self.front.data
        self.front = self.front.next
        if not self.front:
            self.rear = None
        self.size -= 1
        return data
```

### 队列的经典应用

#### 1. 广度优先搜索 (BFS)

```python
from collections import deque

def bfs(graph, start):
    """图的广度优先遍历"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []

    while queue:
        node = queue.popleft()  # 出队
        result.append(node)

        # 将邻居加入队列
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)  # 入队

    return result

# 测试
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print(bfs(graph, 'A'))  # ['A', 'B', 'C', 'D', 'E', 'F']
```

#### 2. 任务调度

```python
from collections import deque

class TaskScheduler:
    def __init__(self):
        self.queue = deque()

    def add_task(self, task):
        """添加任务"""
        self.queue.append(task)
        print(f"任务 '{task}' 已加入队列")

    def process_next(self):
        """处理下一个任务"""
        if not self.queue:
            print("没有待处理的任务")
            return
        task = self.queue.popleft()
        print(f"正在处理任务 '{task}'")
        return task

# 使用
scheduler = TaskScheduler()
scheduler.add_task("发送邮件")
scheduler.add_task("生成报告")
scheduler.add_task("备份数据")

scheduler.process_next()  # 发送邮件（先加入的先处理）
scheduler.process_next()  # 生成报告
```

#### 3. 打印队列

```python
import time
from collections import deque

class PrintQueue:
    def __init__(self):
        self.queue = deque()

    def add_job(self, document):
        self.queue.append(document)
        print(f"文档 '{document}' 加入打印队列")

    def print_jobs(self):
        while self.queue:
            doc = self.queue.popleft()
            print(f"正在打印: {doc}")
            time.sleep(1)  # 模拟打印时间
        print("所有打印任务完成")
```

---

## 🔄 双端队列 (Deque)

### 什么是双端队列？

**两端都可以插入和删除的队列**

```
双端队列：
Front ⇄ [1][2][3][4] ⇄ Rear
↑                      ↑
可以从这里              也可以从这里
插入/删除              插入/删除
```

### Python实现

```python
from collections import deque

dq = deque()

# 两端都可以操作
dq.append(1)        # 右端添加
dq.appendleft(0)    # 左端添加
dq.extend([2, 3])   # 右端批量添加

# [0, 1, 2, 3]

dq.pop()            # 右端移除 → 3
dq.popleft()        # 左端移除 → 0

# [1, 2]
```

### 应用：滑动窗口

```python
from collections import deque

def max_sliding_window(nums, k):
    """找出滑动窗口的最大值"""
    if not nums:
        return []

    dq = deque()  # 存储索引
    result = []

    for i, num in enumerate(nums):
        # 移除超出窗口的元素
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # 移除比当前元素小的元素
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # 窗口形成后记录最大值
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# 测试
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# [3, 3, 5, 5, 6, 7]
```

---

## ⭐ 优先队列 (Priority Queue)

### 什么是优先队列？

**按优先级出队，而不是先进先出**

```
普通队列：    [1][2][3][4] → 先进先出
优先队列：    [4][2][3][1] → 按优先级（如数值大小）
```

### Python实现（使用heapq）

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        """添加元素（优先级越小越先出队）"""
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        """弹出优先级最高的元素"""
        if not self.heap:
            raise IndexError("队列为空")
        return heapq.heappop(self.heap)[1]

    def is_empty(self):
        return len(self.heap) == 0

# 使用
pq = PriorityQueue()
pq.push("任务A", 3)
pq.push("任务B", 1)  # 优先级最高
pq.push("任务C", 2)

print(pq.pop())  # 任务B（优先级1）
print(pq.pop())  # 任务C（优先级2）
print(pq.pop())  # 任务A（优先级3）
```

### 应用：Dijkstra最短路径

```python
import heapq

def dijkstra(graph, start):
    """使用优先队列实现Dijkstra算法"""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (距离, 节点)

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
```

---

## 📊 复杂度对比

| 操作 | 栈 | 队列(数组) | 队列(链表) | 双端队列 | 优先队列 |
|-----|---|----------|----------|---------|---------|
| **Push/Enqueue** | O(1) | O(1) | O(1) | O(1) | O(log n) |
| **Pop/Dequeue** | O(1) | O(n) | O(1) | O(1) | O(log n) |
| **Peek/Front** | O(1) | O(1) | O(1) | O(1) | O(1) |
| **空间** | O(n) | O(n) | O(n) | O(n) | O(n) |

---

## 🎯 选择指南

### 用栈当：
- ✅ 需要LIFO顺序
- ✅ 递归改迭代
- ✅ 撤销操作
- ✅ 括号匹配

### 用队列当：
- ✅ 需要FIFO顺序
- ✅ BFS遍历
- ✅ 任务调度
- ✅ 消息传递

### 用双端队列当：
- ✅ 需要两端操作
- ✅ 滑动窗口
- ✅ 回文检查

### 用优先队列当：
- ✅ 按优先级处理
- ✅ 找Top K
- ✅ 最短路径算法
- ✅ 任务调度（带优先级）

---

## 🔗 相关概念

- [数组与列表](arrays-lists.md) - 栈和队列的底层实现
- [树与图](trees-graphs.md) - BFS使用队列
- [算法基础](../algorithms/) - DFS用栈，BFS用队列

---

**记住**：
1. 栈 = 后进先出 = 递归、撤销
2. 队列 = 先进先出 = 任务调度、BFS
3. Python用deque实现高效队列
4. 优先队列用于按优先级处理
5. 选择合适的数据结构让代码更简洁
