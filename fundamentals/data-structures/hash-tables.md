# Hash Tables - 哈希表

> 快速查找的利器：O(1)的平均查找时间

## 🎯 什么是哈希表？

**哈希表（Hash Table）** = 通过哈希函数将键映射到数组索引的数据结构

```
键 → 哈希函数 → 索引 → 值

"apple" → hash("apple") → 3 → "苹果"
"banana" → hash("banana") → 7 → "香蕉"
```

### 核心思想

直接通过键计算出存储位置，而不需要遍历查找！

---

## 🔑 哈希函数 (Hash Function)

### 什么是哈希函数？

**将任意大小的数据映射到固定大小的值**

```python
def simple_hash(key, table_size):
    """简单的哈希函数示例"""
    return hash(key) % table_size

# 例子
table_size = 10
print(simple_hash("apple", table_size))   # 3
print(simple_hash("banana", table_size))  # 7
print(simple_hash("cherry", table_size))  # 1
```

### 好的哈希函数特性

✅ **确定性** - 相同输入总是产生相同输出
✅ **均匀分布** - 减少冲突
✅ **快速计算** - O(1)时间
✅ **雪崩效应** - 输入微小变化导致输出剧烈变化

### 常见哈希函数

#### 1. 除留余数法

```python
def hash_mod(key, size):
    return key % size

# 适合：整数键
hash_mod(12345, 100)  # 45
```

#### 2. 乘法哈希

```python
def hash_multiply(key, size):
    A = 0.6180339887  # 黄金分割比例
    return int(size * ((key * A) % 1))
```

#### 3. 字符串哈希

```python
def hash_string(s, size):
    """多项式哈希"""
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 31 + ord(char)) % size
    return hash_value

hash_string("hello", 100)
```

### Python的内置hash()

```python
# Python为大多数对象提供了内置hash函数
print(hash("apple"))      # 整数哈希值
print(hash(42))
print(hash((1, 2, 3)))    # 元组可哈希

# 可变对象不可哈希
# print(hash([1, 2, 3]))  # ❌ TypeError: unhashable type: 'list'
```

---

## ⚔️ 哈希冲突 (Hash Collision)

### 什么是冲突？

**不同的键映射到相同的索引**

```
"apple" → hash → 3
"grape" → hash → 3  ← 冲突！
```

### 为什么会冲突？

- 键的数量 > 数组大小
- 哈希函数不够好
- 生日悖论：冲突比想象中更容易发生

---

## 🛠️ 冲突解决方法

### 1. 链地址法 (Chaining)

**每个位置存储一个链表**

```
哈希表：
0: []
1: []
2: []
3: ["apple", "苹果"] → ["grape", "葡萄"]
4: []
5: ["banana", "香蕉"]
```

#### Python实现

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]  # 每个位置是一个列表

    def _hash(self, key):
        """哈希函数"""
        return hash(key) % self.size

    def put(self, key, value):
        """插入键值对：O(1)平均"""
        index = self._hash(key)
        # 检查键是否已存在
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)  # 更新
                return
        # 不存在则添加
        self.table[index].append((key, value))

    def get(self, key):
        """查找：O(1)平均"""
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        raise KeyError(key)

    def delete(self, key):
        """删除：O(1)平均"""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return
        raise KeyError(key)

# 使用
ht = HashTable()
ht.put("apple", "苹果")
ht.put("banana", "香蕉")
ht.put("grape", "葡萄")

print(ht.get("apple"))   # 苹果
print(ht.get("banana"))  # 香蕉

ht.delete("apple")
# print(ht.get("apple"))  # KeyError
```

**优点**：
- ✅ 简单直观
- ✅ 删除容易
- ✅ 负载因子可以 > 1

**缺点**：
- ❌ 额外的指针空间
- ❌ 缓存性能差

---

### 2. 开放寻址法 (Open Addressing)

**冲突时在数组中寻找下一个空位**

#### 2.1 线性探测 (Linear Probing)

```python
# 如果位置i被占用，尝试i+1, i+2, i+3...
index = hash(key) % size
if table[index] is occupied:
    index = (index + 1) % size  # 线性查找下一个
```

```python
class LinearProbingHashTable:
    def __init__(self, size=10):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        """插入"""
        index = self._hash(key)

        # 线性探测找空位
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value  # 更新
                return
            index = (index + 1) % self.size  # 下一个位置

        self.keys[index] = key
        self.values[index] = value

    def get(self, key):
        """查找"""
        index = self._hash(key)

        # 线性探测查找
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size

        raise KeyError(key)
```

**问题**：聚集 (Clustering)
```
连续占用的位置越来越多，形成"堵塞"
[_][_][X][X][X][X][_][_]
        ↑ 聚集区域
```

#### 2.2 二次探测 (Quadratic Probing)

```python
# 尝试 i, i+1², i+2², i+3²...
index = (hash(key) + i * i) % size
```

减少了主聚集问题

#### 2.3 双重哈希 (Double Hashing)

```python
# 使用第二个哈希函数确定步长
index = (hash1(key) + i * hash2(key)) % size
```

最好的开放寻址方法

---

## 📊 负载因子 (Load Factor)

### 什么是负载因子？

```
负载因子 = 元素数量 / 数组大小
α = n / m
```

### 影响

- **α < 0.5**: 空间浪费，但性能好
- **α ≈ 0.75**: 平衡点（Python dict的默认值）
- **α > 1**: 链地址法可以，开放寻址法不行

### 动态调整大小

```python
class DynamicHashTable:
    def __init__(self):
        self.size = 8
        self.count = 0
        self.table = [[] for _ in range(self.size)]

    def put(self, key, value):
        # 检查负载因子
        if self.count / self.size > 0.75:
            self._resize()

        # 插入...
        self.count += 1

    def _resize(self):
        """扩容：通常翻倍"""
        old_table = self.table
        self.size *= 2
        self.table = [[] for _ in range(self.size)]
        self.count = 0

        # 重新哈希所有元素
        for bucket in old_table:
            for key, value in bucket:
                self.put(key, value)
```

---

## 📚 哈希表的变体

### 1. 哈希集合 (Hash Set)

**只存储键，不存储值**

```python
# Python的set
s = set()
s.add("apple")
s.add("banana")

print("apple" in s)  # True, O(1)
print("cherry" in s) # False, O(1)

# 去重
numbers = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(numbers))  # [1, 2, 3, 4]
```

### 2. 哈希映射 (Hash Map)

**存储键值对**

```python
# Python的dict
d = {}
d["apple"] = "苹果"
d["banana"] = "香蕉"

print(d["apple"])  # 苹果, O(1)
print("cherry" in d)  # False, O(1)
```

### 3. 有序哈希表

**保持插入顺序**

```python
# Python 3.7+ dict保持插入顺序
from collections import OrderedDict

od = OrderedDict()
od["c"] = 3
od["a"] = 1
od["b"] = 2

for key in od:
    print(key)  # c, a, b (插入顺序)
```

---

## 🎯 实际应用

### 1. 词频统计

```python
def word_frequency(text):
    freq = {}
    for word in text.split():
        freq[word] = freq.get(word, 0) + 1
    return freq

text = "apple banana apple cherry banana apple"
print(word_frequency(text))
# {'apple': 3, 'banana': 2, 'cherry': 1}
```

### 2. 两数之和

```python
def two_sum(nums, target):
    """找出和为target的两个数"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return None

print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```

### 3. 最长无重复子串

```python
def longest_unique_substring(s):
    """找最长无重复字符子串"""
    seen = {}
    start = max_len = 0

    for i, char in enumerate(s):
        if char in seen and seen[char] >= start:
            start = seen[char] + 1
        else:
            max_len = max(max_len, i - start + 1)
        seen[char] = i

    return max_len

print(longest_unique_substring("abcabcbb"))  # 3 (abc)
```

### 4. 缓存实现 (LRU)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        # 移到末尾（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # 删除最久未使用的（开头）
            self.cache.popitem(last=False)

# 使用
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # 删除key 2
print(cache.get(2))  # -1 (被删除了)
```

### 5. 数据库索引

```python
# 简化的数据库索引
class SimpleIndex:
    def __init__(self):
        self.index = {}  # 哈希索引

    def insert(self, key, record):
        if key not in self.index:
            self.index[key] = []
        self.index[key].append(record)

    def search(self, key):
        """O(1)查找"""
        return self.index.get(key, [])

# 使用
index = SimpleIndex()
index.insert("user_id_123", {"name": "Alice", "age": 30})
index.insert("user_id_456", {"name": "Bob", "age": 25})

print(index.search("user_id_123"))
```

---

## ⏱️ 时间复杂度

| 操作 | 平均 | 最坏 |
|-----|------|------|
| **查找** | O(1) | O(n) |
| **插入** | O(1) | O(n) |
| **删除** | O(1) | O(n) |
| **空间** | O(n) | O(n) |

**最坏情况**：所有键都冲突，退化为链表

---

## 💡 使用建议

### ✅ 适合用哈希表

- 需要快速查找
- 键值对映射
- 去重
- 计数/频率统计
- 缓存

### ❌ 不适合用哈希表

- 需要有序遍历（用树）
- 需要范围查询（用树）
- 键会频繁变化（重新哈希）
- 内存受限（哈希表需要额外空间）

---

## 🔗 相关概念

- [数组与列表](arrays-lists.md) - 哈希表的底层实现
- [树与图](trees-graphs.md) - 有序查找用树
- [算法基础](../algorithms/) - 哈希表在算法中的应用

---

## 📚 Python的dict实现

Python的dict是高度优化的哈希表：

```python
# Python dict特性
d = {"a": 1, "b": 2}

# O(1)操作
d["c"] = 3      # 插入
value = d["a"]  # 查找
del d["b"]      # 删除
"a" in d        # 成员检查

# 保持插入顺序（Python 3.7+）
# 动态调整大小
# 优化的哈希函数
```

---

**记住**：
1. 哈希表 = 空间换时间的典范
2. 平均O(1)查找，但最坏O(n)
3. 好的哈希函数很重要
4. 负载因子影响性能
5. Python的dict和set都是哈希表
6. 适合快速查找，不适合有序遍历
