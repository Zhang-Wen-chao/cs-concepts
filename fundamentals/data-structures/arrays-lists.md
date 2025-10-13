# Arrays and Lists - 数组与列表

> 最基础、最常用的数据结构：如何存储一组有序的数据？

## 🎯 核心概念

**数组和列表都用来存储有序的元素集合，但实现方式不同。**

---

## 📊 数组 (Array)

### 什么是数组？

**连续内存中存储相同类型元素的集合**

```
内存示意图：
地址:  100   104   108   112   116
数据:  [10] [20] [30] [40] [50]
       ↑
    起始地址
```

### 特点

✅ **随机访问O(1)** - 通过索引直接计算地址
✅ **内存连续** - 缓存友好，访问速度快
✅ **固定大小** - 创建时确定大小（C/Java）
❌ **插入删除慢** - 需要移动元素
❌ **大小固定** - 不能动态增长（传统数组）

### Python中的数组

```python
# Python的list其实是动态数组
numbers = [10, 20, 30, 40, 50]

# 访问：O(1)
print(numbers[2])  # 30

# 修改：O(1)
numbers[2] = 35

# 末尾添加：O(1)摊销
numbers.append(60)

# 中间插入：O(n)
numbers.insert(2, 25)  # 需要移动后面所有元素

# 删除：O(n)
numbers.pop(2)  # 需要移动后面所有元素
```

### 其他语言的数组

```java
// Java - 固定大小数组
int[] numbers = new int[5];  // 大小固定
numbers[0] = 10;
numbers[1] = 20;

// 不能改变大小
// numbers[5] = 30;  // ❌ ArrayIndexOutOfBoundsException
```

```c
// C - 静态数组
int numbers[5] = {10, 20, 30, 40, 50};
printf("%d\n", numbers[2]);  // 30

// 动态数组（需要手动管理）
int* dynamic_array = (int*)malloc(5 * sizeof(int));
// 使用后需要释放
free(dynamic_array);
```

### 为什么访问是O(1)？

```python
# 数组访问的本质：地址计算
# arr[i] = 起始地址 + i * 元素大小

# 例如：
# 起始地址 = 100
# 元素大小 = 4字节
# arr[3] 的地址 = 100 + 3 * 4 = 112

# 这是简单的算术运算，所以是O(1)
```

---

## 📈 动态数组 (Dynamic Array)

### 什么是动态数组？

**可以自动增长的数组**（Python的list，Java的ArrayList）

```python
# Python的list是动态数组
arr = []
arr.append(1)  # 自动增长
arr.append(2)
arr.append(3)
# ... 可以无限添加
```

### 工作原理

```python
# 简化的动态数组实现
class DynamicArray:
    def __init__(self):
        self.capacity = 2  # 初始容量
        self.size = 0      # 当前元素数量
        self.array = [None] * self.capacity

    def append(self, item):
        # 如果满了，扩容
        if self.size == self.capacity:
            self._resize()

        self.array[self.size] = item
        self.size += 1

    def _resize(self):
        # 通常扩容为2倍
        self.capacity *= 2
        new_array = [None] * self.capacity

        # 复制所有元素到新数组
        for i in range(self.size):
            new_array[i] = self.array[i]

        self.array = new_array

# 使用
arr = DynamicArray()
arr.append(1)  # 容量2，大小1
arr.append(2)  # 容量2，大小2
arr.append(3)  # 触发扩容！容量4，大小3
```

### 扩容过程可视化

```
初始: 容量=2
[1, 2]
     ↑ 满了

添加3时扩容:
1. 创建新数组，容量=4
   [_, _, _, _]

2. 复制旧元素
   [1, 2, _, _]

3. 添加新元素
   [1, 2, 3, _]
```

### 为什么append是O(1)摊销？

```python
# 假设扩容策略：容量翻倍
# 插入n个元素的总成本：

# 扩容次数：log(n)
# 每次扩容复制的元素：1, 2, 4, 8, ..., n/2
# 总复制成本：1 + 2 + 4 + ... + n/2 = n - 1

# 平均每次插入成本：(n + (n-1)) / n ≈ 2 = O(1)
# 这就是"摊销"的含义
```

---

## 🔗 链表 (Linked List)

### 什么是链表？

**通过指针连接的节点序列**

```
链表结构：
[10 | •]→[20 | •]→[30 | •]→[40 | ∅]
 数据 指针  数据 指针  数据 指针  数据 空

每个节点包含：
- 数据
- 指向下一个节点的指针
```

### 特点

✅ **灵活插入删除O(1)** - 只需改指针
✅ **动态大小** - 无需预分配
❌ **访问慢O(n)** - 必须从头遍历
❌ **额外空间** - 存储指针开销
❌ **缓存不友好** - 内存不连续

### Python实现

```python
# 节点类
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# 链表类
class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        """在末尾添加节点"""
        new_node = Node(data)

        if not self.head:
            self.head = new_node
            return

        # 遍历到最后
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def insert_at_beginning(self, data):
        """在开头插入：O(1)"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        """删除第一个匹配的节点"""
        if not self.head:
            return

        # 如果要删除头节点
        if self.head.data == data:
            self.head = self.head.next
            return

        # 查找要删除的节点
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next

    def display(self):
        """显示所有元素"""
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# 使用
ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.display()  # 10 -> 20 -> 30 -> None

ll.insert_at_beginning(5)
ll.display()  # 5 -> 10 -> 20 -> 30 -> None

ll.delete(20)
ll.display()  # 5 -> 10 -> 30 -> None
```

### 插入操作可视化

```python
# 在中间插入25（在20后面）

# 1. 创建新节点
[25 | ∅]

# 2. 新节点指向20的下一个节点
[10]→[20]→[30]
         ↓
        [25]

# 3. 20指向新节点
[10]→[20]→[25]→[30]

# 只需要改两个指针！O(1)
```

### 删除操作可视化

```python
# 删除20

# 1. 找到20的前一个节点（10）
[10]→[20]→[30]
 ↑    要删除

# 2. 10的next指向20的next
[10]────→[30]
    [20] (游离)

# 只需要改一个指针！O(1)
```

---

## 🔄 双向链表 (Doubly Linked List)

### 什么是双向链表？

**每个节点有两个指针：prev和next**

```
双向链表：
∅←[10 | •]⇄[20 | •]⇄[30 | •]→∅
   prev next prev next prev next
```

### Python实现

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = DNode(data)

        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next

        current.next = new_node
        new_node.prev = current  # 设置反向指针

    def display_forward(self):
        """正向遍历"""
        current = self.head
        while current:
            print(current.data, end=" <-> ")
            current = current.next
        print("None")

    def display_backward(self):
        """反向遍历"""
        if not self.head:
            return

        # 先到末尾
        current = self.head
        while current.next:
            current = current.next

        # 反向遍历
        while current:
            print(current.data, end=" <-> ")
            current = current.prev
        print("None")
```

### 优势

✅ **双向遍历** - 可以向前或向后
✅ **删除更简单** - 不需要记录前一个节点

```python
# 单向链表删除需要前一个节点
# 双向链表可以直接删除
def delete_node(node):
    if node.prev:
        node.prev.next = node.next
    if node.next:
        node.next.prev = node.prev
```

---

## 📊 数组 vs 链表对比

| 特性 | 数组 | 链表 |
|-----|------|------|
| **内存** | 连续 | 分散 |
| **大小** | 固定/动态 | 动态 |
| **访问** | O(1) | O(n) |
| **搜索** | O(n) | O(n) |
| **插入开头** | O(n) | O(1) |
| **插入末尾** | O(1)摊销 | O(n)* |
| **删除** | O(n) | O(1)** |
| **空间开销** | 低 | 高（指针） |
| **缓存性能** | 好 | 差 |

\* 如果有tail指针则O(1)
\** 如果已经找到节点

---

## 🎯 使用场景

### 用数组当：
✅ 需要频繁随机访问
✅ 大小相对固定
✅ 内存连续性重要（性能）
✅ 遍历操作多

**例子**：
- 存储固定大小的数据
- 实现栈、队列
- 数值计算

### 用链表当：
✅ 频繁插入删除
✅ 大小动态变化
✅ 不需要随机访问
✅ 内存碎片化可以接受

**例子**：
- 实现栈、队列、双端队列
- LRU缓存
- 邻接表（图）

---

## 💡 实际应用

### Python的list

```python
# Python的list是动态数组
# 适合：
arr = [1, 2, 3]
arr.append(4)      # 快
value = arr[2]     # 快

# 不适合：
arr.insert(0, 0)   # 慢！O(n)
arr.pop(0)         # 慢！O(n)
```

### Python的collections.deque

```python
from collections import deque

# deque是双向链表
# 适合两端操作
dq = deque([1, 2, 3])
dq.appendleft(0)   # 快！O(1)
dq.popleft()       # 快！O(1)
dq.append(4)       # 快！O(1)
dq.pop()           # 快！O(1)

# 不适合：
value = dq[100]    # 慢！O(n)
```

---

## 🔗 相关概念

- [栈与队列](stacks-queues.md) - 基于数组或链表实现
- [哈希表](hash-tables.md) - 使用数组存储
- [算法复杂度](../algorithms/complexity-analysis.md) - 分析操作效率

---

**记住**：
1. 数组 = 连续内存 = 快速访问
2. 链表 = 指针连接 = 灵活插入删除
3. 没有完美的数据结构，只有合适的选择
4. Python的list是动态数组，不是链表！
5. 理解底层实现，才能做出正确选择
