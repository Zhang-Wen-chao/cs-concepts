# 关系模型与 SQL (Relational Model and SQL)

> 关系型数据库的理论基础和实践语言

## 🎯 核心思想

**关系模型**（E.F. Codd, 1970）：用**二维表**组织数据
**SQL**：操作这些表的标准语言

---

## 📖 关系模型基础

### 核心术语
```
关系 (Relation) = 表 (Table)
元组 (Tuple)   = 行 (Row)
属性 (Attribute)= 列 (Column)
域 (Domain)    = 属性取值范围
```

### 键 (Key)

| 类型 | 说明 | 示例 |
|------|------|------|
| 超键 | 唯一标识记录的超集 | {id}, {id, name} |
| 候选键 | 最小的超键 | {id}, {身份证号} |
| 主键 | 选中的候选键 | PRIMARY KEY (id) |
| 外键 | 引用其他表主键 | FOREIGN KEY (dept_id) |

### 完整性约束
- **实体完整性**：主键 ≠ NULL，不重复
- **参照完整性**：外键必须引用已存在的值
- **用户定义完整性**：CHECK、UNIQUE 等

---

## 🔧 关系代数

| 操作 | 符号 | 含义 | SQL 等价 |
|------|------|------|----------|
| 选择 | σ | 按条件选行 | WHERE |
| 投影 | π | 选列 | SELECT col |
| 连接 | ⋈ | 合并两表 | JOIN ... ON |
| 并 | ∪ | 两查询合集 | UNION |
| 差 | − | 在A不在B | EXCEPT / NOT IN |
| 交 | ∩ | 两查询交集 | INTERSECT |

---

## 💻 SQL 执行顺序

```
SELECT DISTINCT col, AGG(col)  ⑤
FROM t1 JOIN t2 ON cond         ①
WHERE cond                      ②
GROUP BY col                    ③
HAVING cond                     ④
ORDER BY col                    ⑥
LIMIT n OFFSET m                ⑦
```

---

## 📐 范式理论

```
1NF ⊂ 2NF ⊂ 3NF ⊂ BCNF ⊂ 4NF ⊂ 5NF
(每级消除一类冗余，实际常用 3NF / BCNF)
```

| 范式 | 规则 | 消除 |
|------|------|------|
| 1NF | 属性不可再分 | 重复组 |
| 2NF | 完全依赖主键 | 部分依赖 |
| 3NF | 无传递依赖（非键不依赖非键） | 传递依赖 |
| BCNF | 所有决定因素都是候选键 | 更严格的 3NF |
| 4NF | 消除多值依赖 | 独立多值属性 |

### 反范式化
为性能**故意违反范式**：冗余存储数据以减少 JOIN，适用于读多写少场景。

---

## 🎯 最佳实践速查

- **用自增主键**，不用 UUID（性能差）
- **索引高频查询列**，避免在索引列用函数
- **SELECT 具体列**而非 `*`
- **短事务**，勿在事务内做网络请求
- **按相同顺序访问资源**（避免死锁）

---

## 🔗 与其他概念的联系

- **数据结构 (B+树/哈希表)** → `fundamentals/data-structures/`
- **算法 (DP/连接算法)** → `fundamentals/algorithms/`
- **索引** → [indexing-query-optimization.md](./indexing-query-optimization.md)
- **事务** → [transactions-concurrency.md](./transactions-concurrency.md)
- **存储引擎** → [storage-engines.md](./storage-engines.md)

> 详细 SQL 查询示例详见 [practices/systems/databases/](../../practices/systems/databases/)
