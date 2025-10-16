# 索引与查询优化 (Indexing and Query Optimization)

> 如何让数据库查询跑得飞快

## 🎯 核心概念

**索引 = 数据库的"目录"，帮助快速定位数据**

### 关键问题
- 没有索引会怎样？
- 索引是如何工作的？
- 如何选择合适的索引？
- 如何优化慢查询？

---

## 1️⃣ 为什么需要索引？

### 没有索引的查询

```sql
-- 表：users (1,000,000 行)
SELECT * FROM users WHERE id = 12345;

没有索引：
┌─────────────────────────────────────────┐
│  全表扫描 (Full Table Scan)             │
│                                         │
│  从第一行开始，逐行检查：               │
│    Row 1: id = 1      ❌               │
│    Row 2: id = 2      ❌               │
│    ...                                  │
│    Row 12345: id = 12345  ✅ 找到！    │
│    ...                                  │
│    Row 1000000: id = 1000000  (还要继续)│
│                                         │
│  平均需要扫描: 500,000 行               │
│  时间复杂度: O(n)                       │
└─────────────────────────────────────────┘

有索引：
┌─────────────────────────────────────────┐
│  索引查找 (Index Seek)                  │
│                                         │
│  通过 B+ 树索引：                        │
│    1. 查找根节点                        │
│    2. 二分查找到叶子节点                │
│    3. 直接定位到 id = 12345             │
│                                         │
│  需要访问: ~3-4 个节点                  │
│  时间复杂度: O(log n)                   │
└─────────────────────────────────────────┘

性能对比（100万行）：
  全表扫描: ~1000 ms
  索引查找: ~1 ms
  提升: 1000 倍！
```

---

## 2️⃣ B+ 树索引

### 为什么用 B+ 树？

```
数据库索引的特点：
  • 数据量大（百万、千万行）
  • 数据存储在磁盘
  • 磁盘 I/O 很慢（5-10ms 一次）
  • 需要减少磁盘访问次数

B+ 树的优势：
  ✅ 树的高度低（3-4 层）
  ✅ 每次磁盘 I/O 读取多个键
  ✅ 叶子节点链接，支持范围查询
  ✅ 所有数据都在叶子节点，内部节点只存键
```

### B+ 树结构

```
示例：索引 users.id

                     [50, 100]              ← 根节点（内部节点）
                    /    |    \
                   /     |     \
          [10,25,40]  [60,75,90]  [110,125]  ← 内部节点
           /   |   \    /  |  \     /   |
          /    |    \  /   |   \   /    |
    [1-10] [11-25] [26-40] ...       [126-150] ← 叶子节点
      ↓      ↓       ↓                   ↓
    数据    数据     数据                数据
      ↔      ↔       ↔                   ↔
    (叶子节点之间有双向链表)

特点：
  • 内部节点：只存键（用于导航）
  • 叶子节点：存键+数据指针
  • 叶子节点链接：支持范围扫描
  • 每个节点大小 = 磁盘页大小（4KB/16KB）
```

### 查询过程

```sql
SELECT * FROM users WHERE id = 75;

步骤：
1. 读取根节点 [50, 100]
   → 75 在 50 和 100 之间
   → 跳转到中间子节点

2. 读取内部节点 [60, 75, 90]
   → 75 等于第二个键
   → 跳转到对应叶子节点

3. 读取叶子节点，找到 id=75 的数据指针
   → 根据指针读取实际数据

总共磁盘 I/O：3-4 次
时间：~15-20ms（假设每次 I/O 5ms）
```

### 范围查询

```sql
SELECT * FROM users WHERE id BETWEEN 60 AND 90;

步骤：
1. 通过 B+ 树找到起始位置（id=60 的叶子节点）
2. 顺着叶子节点链表扫描到 id=90
   → 叶子节点是有序的！
   → 不需要再次搜索

这就是 B+ 树的优势：范围查询高效！
```

---

## 3️⃣ 索引类型

### 1. 聚簇索引 (Clustered Index)

```
数据按索引顺序物理存储

InnoDB 的主键就是聚簇索引：
┌─────────────────────────────────────┐
│  B+ 树叶子节点直接存储完整行数据    │
│                                     │
│    [键: id=1, 数据: {id:1, name:Alice, age:20}]
│    [键: id=2, 数据: {id:2, name:Bob, age:22}]
│    [键: id=3, 数据: {id:3, name:Carol, age:21}]
│                                     │
└─────────────────────────────────────┘

特点：
  • 一个表只能有一个聚簇索引（主键）
  • 数据物理有序
  • 查询主键最快（不需要回表）
  • 插入可能导致页分裂（影响性能）
```

### 2. 非聚簇索引 (Secondary Index)

```
索引和数据分离存储

CREATE INDEX idx_name ON users(name);

┌─────────────────────────────────────┐
│  B+ 树叶子节点存储：键 + 主键值     │
│                                     │
│    [键: "Alice", 主键: 1]          │
│    [键: "Bob",   主键: 2]          │
│    [键: "Carol", 主键: 3]          │
│                                     │
└─────────────────────────────────────┘

查询过程：
  SELECT * FROM users WHERE name = 'Bob';

  1. 在 idx_name 索引中找到 "Bob" → 得到主键 2
  2. 用主键 2 去聚簇索引查找完整数据
     → 这叫"回表" (Table Lookup)

  两次索引查询！
```

### 3. 覆盖索引 (Covering Index)

```
索引包含查询所需的所有列

CREATE INDEX idx_name_age ON users(name, age);

SELECT name, age FROM users WHERE name = 'Bob';
                   ↑
        只需要 name 和 age，索引已经包含！
        → 不需要回表，直接从索引返回

性能提升：避免了回表的开销
```

### 4. 联合索引 (Composite Index)

```
多列组成的索引

CREATE INDEX idx_name_age ON users(name, age);

索引结构（按 name, age 排序）：
  [Alice, 20] → 主键 1
  [Alice, 25] → 主键 4
  [Bob, 22]   → 主键 2
  [Carol, 21] → 主键 3

最左前缀原则：
  ✅ WHERE name = 'Bob'                (用到索引)
  ✅ WHERE name = 'Bob' AND age = 22   (用到索引)
  ❌ WHERE age = 22                    (用不到索引)

  原因：索引先按 name 排序，再按 age 排序
        直接查 age 无法利用索引的有序性
```

---

## 4️⃣ 哈希索引

### 原理

```
使用哈希表存储索引

CREATE INDEX idx_hash ON users(id) USING HASH;

┌──────────────────────────────────┐
│  Hash(id) → 数据指针             │
│                                  │
│  Hash(1) = 123   → 指针A        │
│  Hash(2) = 456   → 指针B        │
│  Hash(3) = 789   → 指针C        │
│                                  │
└──────────────────────────────────┘

优点：
  • 等值查询极快 O(1)
  • SELECT * FROM users WHERE id = 5;

缺点：
  ❌ 不支持范围查询
  ❌ 不支持排序
  ❌ 不支持最左前缀
  ❌ 哈希冲突处理开销
```

### 适用场景

```
使用哈希索引：
  • 只有等值查询
  • 内存数据库（Redis）

使用 B+ 树索引：
  • 范围查询
  • 排序
  • 大多数关系型数据库
```

---

## 5️⃣ 索引设计原则

### 1. 选择性高的列

```
选择性 = 不同值数量 / 总行数

例子：users 表（100万行）
  • id 列：1,000,000 种不同值 → 选择性 = 1.0  ✅
  • age 列：100 种不同值       → 选择性 = 0.0001  ⚠️
  • gender: 2 种不同值         → 选择性 = 0.000002  ❌

选择性越高，索引越有效！

gender 列建索引几乎没用：
  SELECT * FROM users WHERE gender = 'M';
  → 会返回约 50% 的数据
  → 还不如全表扫描
```

### 2. 频繁查询的列

```
在 WHERE、JOIN、ORDER BY、GROUP BY 中常用的列建索引

✅ WHERE id = 123          → id 建索引
✅ WHERE age > 20          → age 建索引
✅ ORDER BY create_time    → create_time 建索引
✅ JOIN ON user_id         → user_id 建索引
```

### 3. 避免过多索引

```
索引的代价：
  • 占用磁盘空间
  • 插入/更新/删除时需要维护索引
  • 查询优化器选择索引的时间增加

例子：
  users 表有 10 个索引
  INSERT 一行数据 →需要更新 10 个索引
  → 插入变慢 10 倍！

建议：
  • 只为常用查询建索引
  • 定期检查未使用的索引并删除
```

### 4. 联合索引顺序

```
根据查询模式设计顺序

场景1：查询条件固定
  WHERE name = ? AND age = ?
  → CREATE INDEX idx ON users(name, age);

场景2：独立查询也很多
  WHERE name = ?    (频繁)
  WHERE age = ?     (频繁)
  WHERE name = ? AND age = ?  (频繁)

  → 建两个索引:
    INDEX idx_name ON users(name)
    INDEX idx_age ON users(age)

场景3：选择性考虑
  WHERE country = ? AND city = ?
  国家选择性低（200个），城市选择性高（10000个）

  ✅ INDEX(country, city) → 合理
     先过滤国家，再精确定位城市

  ❌ INDEX(city, country) → 浪费
     city 已经很精确，country 帮助不大
```

---

## 6️⃣ 查询优化

### 使用 EXPLAIN 分析查询

```sql
EXPLAIN SELECT * FROM users WHERE age > 20;

结果：
┌────┬─────────────┬──────┬──────┬─────────┬──────┬───────┬─────┐
│ id │ select_type │table │ type │  key    │ rows │ Extra │ ... │
├────┼─────────────┼──────┼──────┼─────────┼──────┼───────┼─────┤
│ 1  │  SIMPLE     │users │ ALL  │  NULL   │100000│ Using │ ... │
│    │             │      │      │         │      │ where │     │
└────┴─────────────┴──────┴──────┴─────────┴──────┴───────┴─────┘

关键字段：
  • type: 访问类型
    - ALL: 全表扫描 ❌ 最差
    - index: 索引扫描
    - range: 范围扫描 ✅ 较好
    - ref: 索引查找 ✅ 好
    - eq_ref: 唯一索引查找
    - const: 常量查找 ✅ 最好

  • key: 使用的索引
    - NULL: 没有使用索引 ❌

  • rows: 扫描的行数
    - 越少越好

  • Extra:
    - Using index: 覆盖索引 ✅
    - Using where: 服务器层过滤
    - Using filesort: 文件排序 ❌ 慢
    - Using temporary: 临时表 ❌ 慢
```

### 常见优化技巧

#### 1. 避免 SELECT *

```sql
-- ❌ 不好
SELECT * FROM users WHERE id = 1;

-- ✅ 更好
SELECT id, name, email FROM users WHERE id = 1;

原因：
  • 减少网络传输
  • 可能利用覆盖索引
  • 减少 I/O
```

#### 2. 使用 LIMIT

```sql
-- ❌ 返回所有结果
SELECT * FROM users WHERE age > 20;

-- ✅ 限制结果集
SELECT * FROM users WHERE age > 20 LIMIT 100;

原因：
  • 减少数据传输
  • 提前终止查询
```

#### 3. 避免在索引列上使用函数

```sql
-- ❌ 不会使用索引
SELECT * FROM users WHERE YEAR(birthday) = 2000;

-- ✅ 会使用索引
SELECT * FROM users
WHERE birthday BETWEEN '2000-01-01' AND '2000-12-31';

原因：
  函数破坏了索引的有序性
```

#### 4. 避免隐式类型转换

```sql
-- id 是 INT 类型

-- ❌ 字符串比较，不会使用索引
SELECT * FROM users WHERE id = '123';

-- ✅ 数字比较，会使用索引
SELECT * FROM users WHERE id = 123;
```

#### 5. 使用 IN 代替 OR

```sql
-- ❌ OR 可能不走索引
SELECT * FROM users WHERE id = 1 OR id = 2 OR id = 3;

-- ✅ IN 会优化
SELECT * FROM users WHERE id IN (1, 2, 3);
```

#### 6. 避免前导模糊查询

```sql
-- ❌ 前导通配符不走索引
SELECT * FROM users WHERE name LIKE '%Alice';

-- ✅ 后缀通配符走索引
SELECT * FROM users WHERE name LIKE 'Alice%';

原因：
  B+ 树是按键值有序的
  '%Alice' 无法确定从哪里开始查找
```

---

## 7️⃣ 查询执行计划

### 查询优化器的工作

```
SQL 查询 → 查询优化器 → 执行计划

优化器考虑：
  1. 可用的索引
  2. 统计信息（表大小、索引选择性）
  3. 不同执行方案的代价
  4. 选择代价最小的方案
```

### 连接算法

```
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

1. Nested Loop Join（嵌套循环）
   for each row in users:
       for each row in orders:
           if match: return

   时间复杂度：O(n × m)
   适用：小表

2. Hash Join（哈希连接）
   • 扫描 users，构建哈希表
   • 扫描 orders，探测哈希表

   时间复杂度：O(n + m)
   适用：大表等值连接

3. Merge Join（归并连接）
   • 两表都按连接键排序
   • 同时扫描，合并结果

   时间复杂度：O(n log n + m log m)
   适用：已排序或有索引
```

---

## 8️⃣ 实际案例

### 案例 1：慢查询优化

```sql
-- 原查询：3 秒
SELECT * FROM orders
WHERE user_id = 123
  AND status = 'completed'
  AND create_time > '2024-01-01';

-- 分析
EXPLAIN：type=ALL, rows=1000000

-- 问题：全表扫描，没有索引

-- 解决：创建联合索引
CREATE INDEX idx_user_status_time
ON orders(user_id, status, create_time);

-- 优化后：0.01 秒
-- EXPLAIN：type=range, key=idx_user_status_time, rows=50
```

### 案例 2：分页优化

```sql
-- 深分页很慢：10 秒
SELECT * FROM posts
ORDER BY id DESC
LIMIT 1000000, 20;

-- 问题：需要扫描前 1000000 行

-- 优化：使用游标
SELECT * FROM posts
WHERE id < last_seen_id
ORDER BY id DESC
LIMIT 20;

-- 或者：子查询先找 ID
SELECT * FROM posts
WHERE id IN (
    SELECT id FROM posts
    ORDER BY id DESC
    LIMIT 1000000, 20
);

-- 时间：0.1 秒
```

### 案例 3：COUNT 优化

```sql
-- 慢：2 秒
SELECT COUNT(*) FROM orders WHERE status = 'pending';

-- 问题：全表扫描

-- 优化1：添加索引
CREATE INDEX idx_status ON orders(status);

-- 优化2：使用近似值（适合大表）
SELECT COUNT(*) FROM information_schema.TABLES
WHERE TABLE_NAME = 'orders';

-- 优化3：维护计数器表
CREATE TABLE order_stats (
    status VARCHAR(20),
    count INT
);
-- 用触发器或定时任务更新
```

---

## 🔗 与其他概念的联系

### 与数据结构
- **B+ 树** - 索引的核心实现
- **哈希表** - 哈希索引

参考：`fundamentals/data-structures/`

### 与算法
- **排序算法** - ORDER BY 实现
- **连接算法** - JOIN 优化

参考：`fundamentals/algorithms/`

### 与操作系统
- **缓冲管理** - 减少磁盘 I/O
- **页缓存** - 加速访问

参考：`systems/operating-systems/`

---

## 📚 深入学习

### 推荐资源
- *High Performance MySQL* - 查询优化章节
- *Database System Concepts* - 索引章节
- MySQL/PostgreSQL 官方文档

### 实践项目
- 分析真实应用的慢查询
- 使用 EXPLAIN 优化查询
- 设计合理的索引策略

### 下一步
- [事务与并发控制](./transactions-concurrency.md) - 保证数据一致性
- [存储引擎](./storage-engines.md) - 理解索引实现
- [关系模型](./relational-model-sql.md) - SQL 基础

---

**掌握索引和查询优化，让你的数据库飞起来！** 🚀
