# 索引与查询优化 (Indexing and Query Optimization)

> 如何让数据库查询跑得飞快

## 🎯 核心思想

**索引 = 数据库的"目录"**，通过数据结构加速数据定位。无索引时全表扫描 O(n)，有索引可达 O(log n)。

---

## 🌲 B+ 树索引

### 为什么用 B+ 树？
- 高度低（3-4 层），磁盘 I/O 次数少
- 叶子节点有序链接，支持高效范围查询
- 内部节点只存键，不存数据

```
                     [50,100]             ← 根节点
                    /    |    \
          [10,25,40]  [60,75,90] [110,125]  ← 内部节点
           /   |   \    /  |  \     /   |
    [1-10] [11-25] ...   [60-75] ...    [126-150]  ← 叶子节点(含数据/主键)
      ↔      ↔            ↔               ↔      ← 双向链表
```

### 查询过程
```sql
SELECT * FROM users WHERE id = 75;
-- ① 根节点 → ② 内部节点 → ③ 叶子节点 → ④ 读数据
-- 磁盘 I/O: ~3-4 次
```

### 范围查询
```sql
SELECT * FROM users WHERE id BETWEEN 60 AND 90;
-- 找到起点后沿叶子节点链表顺序扫描
```

---

## 🏷️ 索引类型对比

| 类型 | 特点 | 说明 |
|------|------|------|
| 聚簇索引 (Clustered) | 叶子存完整行；表只能一个 | InnoDB 主键默认 |
| 二级索引 (Secondary) | 叶子存主键值；需回表 | 其他建索引的列 |
| 覆盖索引 (Covering) | 索引包含查询全部列 | 免回表 |
| 联合索引 (Composite) | 多列组合；最左前缀原则 | 顺序 = 查询模式 |
| 哈希索引 (Hash) | 等值 O(1) / 不支持范围 | Memory 引擎 |

### 最左前缀原则
```sql
INDEX(name, age)  -- 按 name 排序，name 相同再按 age
✅ WHERE name = 'Bob'
✅ WHERE name = 'Bob' AND age = 22
❌ WHERE age = 22               -- 跳过了 name
```

---

## 🔍 查询优化利器：EXPLAIN

```
EXPLAIN SELECT * FROM users WHERE age > 20;
```

| 字段 | 含义 | 好/坏 |
|------|------|-------|
| type | 访问方式 | const > ref > range > index > **ALL** |
| key | 使用索引 | NULL = 未使用 ❌ |
| rows | 扫描行数 | 越小越好 |
| Extra | 补充信息 | Using index✅ / filesort❌ |

---

## ⚡ 常见优化技巧

| 原则 | ❌ 坏写法 | ✅ 好写法 |
|------|----------|----------|
| 具体列 | `SELECT *` | `SELECT id, name` |
| 限制结果 | 无 LIMIT | `LIMIT 100` |
| 索引函数 | `WHERE YEAR(birth)=2000` | `WHERE birth BETWEEN '2000-01-01' AND '2000-12-31'` |
| 前导模糊 | `LIKE '%Alice'` | `LIKE 'Alice%'` |
| OR → IN | `id=1 OR id=2` | `id IN (1,2)` |

### 连接算法比较
```
Nested Loop: O(n×m)  → 小表驱动大表
Hash Join:   O(n+m)  → 大表等值连接
Merge Join: O(nlogn) → 已排序/有索引
```

---

## 🔗 与其他概念的联系

- **数据结构 (B+树/哈希表)** → `fundamentals/data-structures/`
- **算法 (排序/DP优化器)** → `fundamentals/algorithms/`
- **OS (缓冲管理/页缓存)** → `systems/operating-systems/`
- **存储引擎** → [storage-engines.md](./storage-engines.md)
- **事务** → [transactions-concurrency.md](./transactions-concurrency.md)

> 详细 EXPLAIN 案例与优化实战详见 [practices/systems/databases/](../../practices/systems/databases/)
