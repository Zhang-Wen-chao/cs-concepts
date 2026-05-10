# 存储引擎 (Storage Engines)

> 数据在磁盘上如何组织和存储的核心机制

## 🎯 核心思想

**存储引擎 = 数据库的"心脏"**，决定了数据如何存储、读取、索引、事务和锁管理。不同引擎有不同的性能特点。

---

## 🗄️ InnoDB（默认引擎，通用场景）

### 存储层次

```
表空间 → 段 → 区 (1MB, 64页) → 页 (16KB, 最小I/O) → 行
```

### 索引组织表 (IOT)

- **聚簇索引**：数据和索引在一起，按主键排序存储
- **二级索引**：叶子存主键值，查询需**回表**
- **覆盖索引**：索引包含全部查询列，免回表

```
聚簇索引 (叶子存整行)        二级索引 (叶子存主键)
  [id=1, {完整行}]            [name="Bob", id=2] → 回表→[id=2, {完整行}]
  [id=2, {完整行}]            [name="Carol", id=3]
  [id=3, {完整行}]            ...
```

### 日志系统

| 日志 | 作用 | 特点 |
|------|------|------|
| **Redo Log** | 持久性 (Durability) | WAL 顺序写，崩溃重演 |
| **Undo Log** | 原子性 + MVCC | 记录旧版本，回滚/快照 |
| **Binlog** | 复制/恢复 | MySQL 层逻辑日志 |

**两阶段提交**：保证 Redo Log 与 Binlog 一致
```
Prepare(REDO prepare) → 写 Binlog → Commit(REDO commit)
崩溃恢复：如果 REDO=prepare 且 Binlog 存在 → 提交；否则回滚
```

### Buffer Pool（缓冲池）

- 缓存数据页/索引页，减少磁盘 I/O
- LRU 管理（新页插 5/8 处，防全表扫描污染）
- 脏页通过 Redo Log checkpoint 控制刷盘

### Change Buffer（写缓冲）

- 优化非唯一二级索引的随机写入
- 写入时索引页不在内存 → 暂存 Change Buffer → 后台 merge

---

## ⚡ MyISAM（只读/归档场景）

| 特性 | InnoDB | MyISAM |
|------|--------|--------|
| 事务 | ✅ | ❌ |
| 锁粒度 | 行锁 | 表锁 |
| MVCC | ✅ | ❌ |
| 崩溃安全 | ✅ | ❌ |
| 全文索引 | ❌ | ✅ |
| 文件 | .ibd（数据+索引） | .frm+.MYD+.MYI |

---

## 🌲 LSM 树引擎（写密集场景）

**RocksDB / LevelDB / HBase** 的核心结构

### 设计思想
```
写入快 (顺序 I/O):
  MemTable(内存) → Immutable → SSTable(Level 0) → Compaction → Level N

读取慢 (需查多层):
  MemTable → Immutable → Level 0 → Level 1 → ...
  优化: Bloom Filter 快速判不存在, Block Cache 缓存热点
```

### B+树 vs LSM 树对比

| 维度 | B+树 (InnoDB) | LSM 树 (RocksDB) |
|------|--------------|------------------|
| 写入 | 随机写，页分裂 | **顺序写**，快 |
| 读取 | 稳定 O(log n) | 需查多文件，慢 |
| 空间 | 利用率较高 | 写放大 |
| 场景 | OLTP 通用 | 写多读少、时序 |

---

## 🔄 行存储 vs 列存储

| 维度 | 行存储 (InnoDB) | 列存储 (ClickHouse) |
|------|-----------------|---------------------|
| 写入 | 一次 I/O 写整行 | 分散到多列文件 |
| 分析查询 | 需扫整行 | 只读目标列 |
| 压缩率 | 低 | 高（同类数据连续） |
| 场景 | OLTP（事务） | OLAP（分析） |

---

## 🎯 最佳实践

- **主键用自增 BIGINT**（顺序插入，避免页分裂）
- `innodb_buffer_pool_size` 设为内存的 50-80%
- `innodb_file_per_table = ON`（独立表空间）
- 只读归档用 Archive 引擎
- LSM 引擎适合写入密集场景（日志、时序）

---

## 🔗 与其他概念的联系

- **数据结构 (B+树/跳表/BloomFilter)** → `fundamentals/data-structures/`
- **OS (页/缓冲池/WAL)** → `systems/operating-systems/`
- **索引** → [indexing-query-optimization.md](./indexing-query-optimization.md)
- **事务** → [transactions-concurrency.md](./transactions-concurrency.md)

> 详细配置与引擎实测详见 [practices/systems/databases/](../../practices/systems/databases/)
