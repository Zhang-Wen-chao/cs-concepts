# 存储引擎 (Storage Engines)

> 数据在磁盘上如何组织和存储的核心机制

## 🎯 核心思想

存储引擎是数据库的"心脏"，决定了数据如何在磁盘上存储、如何被读取和修改。理解存储引擎，就能理解数据库性能的本质。

**一句话理解：**
存储引擎是数据库底层的**存储和检索**实现，不同的引擎有不同的性能特点。

## 📖 存储引擎概述

### 1. 什么是存储引擎？

```
数据库系统的层次结构：

┌─────────────────────────────────┐
│      SQL 接口 (SQL Interface)    │  ← 用户交互
├─────────────────────────────────┤
│    查询优化器 (Query Optimizer)   │  ← 生成执行计划
├─────────────────────────────────┤
│    执行引擎 (Execution Engine)    │  ← 执行查询
├─────────────────────────────────┤
│    存储引擎 (Storage Engine)      │  ← 数据存储和检索
├─────────────────────────────────┤
│       文件系统 (File System)      │  ← 磁盘 I/O
└─────────────────────────────────┘

存储引擎负责：
  • 数据如何存储到磁盘
  • 数据如何从磁盘读取
  • 索引如何组织
  • 事务如何实现
  • 锁如何管理
```

### 2. MySQL 存储引擎

```sql
-- 查看支持的存储引擎
SHOW ENGINES;

主要引擎：
┌──────────┬────────┬─────────┬────────┬──────┐
│ Engine   │ 事务   │ 锁粒度  │ MVCC   │ 场景 │
├──────────┼────────┼─────────┼────────┼──────┤
│ InnoDB   │ 支持   │ 行锁    │ 支持   │ OLTP │
│ MyISAM   │ 不支持 │ 表锁    │ 不支持 │ 只读 │
│ Memory   │ 不支持 │ 表锁    │ 不支持 │ 临时 │
│ Archive  │ 不支持 │ 行锁    │ 不支持 │ 归档 │
└──────────┴────────┴─────────┴────────┴──────┘

-- 创建表时指定引擎
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50)
) ENGINE=InnoDB;

-- 修改引擎
ALTER TABLE users ENGINE=MyISAM;
```

## 🗄️ InnoDB 存储引擎

MySQL 的默认引擎，功能最完善，适合绝大多数场景。

### 1. 存储结构

#### 表空间 (Tablespace)
```
InnoDB 的存储层次：

表空间 (Tablespace)
  ├─ 段 (Segment)
  │   ├─ 数据段 (Leaf Node Segment)
  │   ├─ 索引段 (Non-Leaf Node Segment)
  │   └─ 回滚段 (Rollback Segment)
  │
  └─ 区 (Extent) - 1MB，连续 64 页
      └─ 页 (Page) - 16KB，最小 I/O 单位
          └─ 行 (Row) - 实际数据

磁盘文件：
  • 系统表空间: ibdata1
  • 独立表空间: table_name.ibd (每个表一个文件)
  • Redo 日志: ib_logfile0, ib_logfile1
  • Undo 日志: undo001, undo002
```

#### 页结构 (Page Structure)
```
页是 InnoDB 的基本存储单位，默认 16KB

┌─────────────────────────────────────┐
│        File Header (38 bytes)       │ ← 页头：页号、校验和等
├─────────────────────────────────────┤
│        Page Header (56 bytes)       │ ← 页信息：记录数、空闲空间等
├─────────────────────────────────────┤
│      Infimum + Supremum (26 bytes)  │ ← 虚拟记录：最小和最大
├─────────────────────────────────────┤
│          User Records               │ ← 用户记录（链表）
│                                     │
│          ···                        │
├─────────────────────────────────────┤
│          Free Space                 │ ← 空闲空间
├─────────────────────────────────────┤
│        Page Directory               │ ← 页目录：记录偏移量
├─────────────────────────────────────┤
│        File Trailer (8 bytes)       │ ← 页尾：校验和
└─────────────────────────────────────┘

特点：
  • 页内记录按主键顺序排列
  • 使用链表连接记录
  • Page Directory 用于二分查找
```

#### 行格式 (Row Format)
```
InnoDB 支持多种行格式：

1. COMPACT (默认)
┌────────┬──────────┬─────────┬──────┬─────┐
│ 变长   │ NULL标志 │ 记录头  │ 列1  │ ... │
│ 字段长度│ 位图     │ 信息    │      │     │
└────────┴──────────┴─────────┴──────┴─────┘

2. DYNAMIC (推荐)
  • 长字段（TEXT、BLOB）存储在溢出页
  • 只在行中保留 20 字节指针

3. COMPRESSED
  • 支持压缩，节省空间
  • 增加 CPU 开销

-- 指定行格式
CREATE TABLE users (
    id INT,
    data TEXT
) ENGINE=InnoDB ROW_FORMAT=DYNAMIC;

隐藏列：
  • DB_TRX_ID (6 bytes): 事务 ID
  • DB_ROLL_PTR (7 bytes): 回滚指针
  • DB_ROW_ID (6 bytes): 行 ID（如果没有主键）
```

### 2. 索引组织表 (IOT)

```
InnoDB 使用聚簇索引 (Clustered Index) 组织数据

聚簇索引：数据和索引在一起，按主键顺序存储

┌─────────────────────────────────────┐
│       主键索引 (B+ Tree)             │
│                                     │
│         ┌───┐                       │
│         │ 5 │                       │
│      ┌──┴─┬─┴──┐                    │
│     ┌┴─┐ ┌┴─┐ ┌┴─┐                  │
│     │2 │ │5 │ │8 │                  │
│     └┬─┘ └┬─┘ └┬─┘                  │
│      │    │    │                    │
│   ┌──┴─┐ ┌┴──┐ ┌┴───┐               │
│   │1,2 │ │3,5│ │7,8 │  ← 叶子节点存储完整行数据
│   └────┘ └───┘ └────┘               │
└─────────────────────────────────────┘

优点：
  • 范围查询快（叶子节点有序且连续）
  • 按主键查询直接返回数据
  • 行数据和索引在一起，减少 I/O

缺点：
  • 插入时可能导致页分裂
  • 二级索引需要回表
```

**二级索引 (Secondary Index)**
```
二级索引的叶子节点存储主键值

┌─────────────────────────────────────┐
│      二级索引 (name 字段)            │
│                                     │
│           ┌─────┐                   │
│           │ 李  │                   │
│       ┌───┴──┬──┴───┐               │
│      ┌┴──┐ ┌┴──┐ ┌┴──┐              │
│      │张 │ │李 │ │王 │              │
│      └┬──┘ └┬──┘ └┬──┘              │
│       │     │     │                 │
│   ┌───┴┐  ┌┴───┐ ┌┴────┐            │
│   │张,1│  │李,3│ │王,5 │  ← 存储主键值
│   └────┘  └────┘ └─────┘            │
└─────────────────────────────────────┘
              ↓
          回表 (通过主键索引查找完整数据)
┌─────────────────────────────────────┐
│       主键索引                       │
│         ...                         │
│        │3,李四,25│                   │
│         ...                         │
└─────────────────────────────────────┘

回表过程：
  1. 在二级索引中找到 "李四"，获取主键 3
  2. 用主键 3 在聚簇索引中查找
  3. 返回完整行数据

覆盖索引 (避免回表)：
  -- 如果查询的列都在索引中，不需要回表
  SELECT id, name FROM users WHERE name = '李四';
         ↑     ↑                      ↑
       主键   索引列                 索引列

  不需要回表！
```

### 3. 日志系统

#### Redo Log (重做日志)
```
作用：保证持久性 (Durability)

问题：
  • 修改数据需要随机写磁盘（慢）
  • 事务提交时如何保证数据持久化？

解决：Write-Ahead Logging (WAL)
  1. 先写 Redo Log（顺序写，快）
  2. 再修改内存中的数据页
  3. 后台异步将脏页刷盘

┌─────────────────────────────────────┐
│         Redo Log 流程                │
│                                     │
│  事务修改数据                        │
│      ↓                              │
│  写 Redo Log Buffer                 │
│      ↓                              │
│  写 Redo Log File (磁盘)  ← 顺序写   │
│      ↓                              │
│  修改 Buffer Pool 中的数据页         │
│      ↓                              │
│  后台线程刷脏页到磁盘                │
└─────────────────────────────────────┘

Redo Log 文件：
  • ib_logfile0, ib_logfile1 (循环使用)
  • 固定大小，写满后覆盖旧的
  • 使用 checkpoint 标记哪些日志已刷盘

崩溃恢复：
  1. 从 checkpoint 开始扫描 Redo Log
  2. 重做所有已提交事务的修改
  3. 回滚未提交事务

配置：
  innodb_log_file_size = 512M  # 单个日志文件大小
  innodb_log_buffer_size = 16M # 日志缓冲区大小
  innodb_flush_log_at_trx_commit = 1  # 每次提交都刷盘
```

#### Undo Log (回滚日志)
```
作用：
  1. 保证原子性 (Atomicity) - 事务回滚
  2. 实现 MVCC - 读取旧版本

记录内容：
  • INSERT → 记录主键，回滚时删除
  • DELETE → 记录完整行，回滚时插入
  • UPDATE → 记录旧值，回滚时恢复

┌─────────────────────────────────────┐
│      Undo Log 链表                   │
│                                     │
│  当前版本                            │
│  ┌────────────────┐                 │
│  │ id=1, name=李四│                 │
│  │ trx_id=100     │                 │
│  └────────┬───────┘                 │
│           │ roll_ptr                │
│           ↓                         │
│  ┌────────────────┐                 │
│  │ id=1, name=张三│  ← Undo Log     │
│  │ trx_id=50      │                 │
│  └────────┬───────┘                 │
│           │ roll_ptr                │
│           ↓                         │
│  ┌────────────────┐                 │
│  │ id=1, name=王五│  ← Undo Log     │
│  │ trx_id=20      │                 │
│  └────────────────┘                 │
└─────────────────────────────────────┘

MVCC 读取旧版本：
  • 事务 ID=60 读取：看到 "张三"
  • 事务 ID=110 读取：看到 "李四"

Undo Log 清理：
  • 当没有事务需要旧版本时，可以清理
  • Purge 线程负责清理
```

#### Binlog (二进制日志)
```
作用：
  1. 主从复制
  2. 数据恢复
  3. 审计

与 Redo Log 的区别：
┌──────────────┬────────────┬────────────┐
│              │ Redo Log   │ Binlog     │
├──────────────┼────────────┼────────────┤
│ 层次         │ InnoDB     │ MySQL      │
│ 内容         │ 物理日志   │ 逻辑日志   │
│ 大小         │ 固定       │ 追加       │
│ 作用         │ 崩溃恢复   │ 复制、备份 │
└──────────────┴────────────┴────────────┘

Binlog 格式：
  • Statement: 记录 SQL 语句
    - 优点：日志量小
    - 缺点：不能保证主从一致（如 NOW()）

  • Row: 记录每行的变化
    - 优点：主从一致
    - 缺点：日志量大

  • Mixed: 混合模式
    - 一般用 Statement，必要时用 Row

配置：
  log_bin = /var/lib/mysql/binlog
  binlog_format = ROW
  expire_logs_days = 7  # 保留 7 天
```

#### 两阶段提交 (2PC)
```
为了保证 Redo Log 和 Binlog 的一致性

问题：
  如果先写 Redo Log 再写 Binlog：
    • Redo Log 写成功，Binlog 失败
    • 崩溃恢复后数据存在，但从库没有

  如果先写 Binlog 再写 Redo Log：
    • Binlog 写成功，Redo Log 失败
    • 崩溃恢复后数据不存在，但从库有

解决：两阶段提交
┌─────────────────────────────────────┐
│  1. Prepare 阶段                    │
│     写 Redo Log，标记为 prepare      │
│          ↓                          │
│  2. 写 Binlog                       │
│          ↓                          │
│  3. Commit 阶段                     │
│     写 Redo Log，标记为 commit       │
└─────────────────────────────────────┘

崩溃恢复时：
  • 如果 Redo Log 是 prepare，检查 Binlog：
    - Binlog 存在 → 提交事务
    - Binlog 不存在 → 回滚事务
  • 如果 Redo Log 是 commit → 已提交
```

### 4. Buffer Pool (缓冲池)

```
作用：缓存数据页和索引页，减少磁盘 I/O

┌─────────────────────────────────────┐
│          Buffer Pool                │
│  ┌──────────────────────────┐       │
│  │    数据页缓存             │       │
│  ├──────────────────────────┤       │
│  │    索引页缓存             │       │
│  ├──────────────────────────┤       │
│  │    Insert Buffer         │       │
│  ├──────────────────────────┤       │
│  │    自适应哈希索引         │       │
│  ├──────────────────────────┤       │
│  │    锁信息                │       │
│  └──────────────────────────┘       │
└─────────────────────────────────────┘

页的管理：
  • LRU List: 最近最少使用
    - 新读取的页放在 midpoint (默认 5/8 处)
    - 避免全表扫描污染缓存

  • Free List: 空闲页

  • Flush List: 脏页 (需要刷盘的页)

刷脏页时机：
  1. Redo Log 写满
  2. Buffer Pool 空间不足
  3. MySQL 空闲时
  4. MySQL 正常关闭

配置：
  innodb_buffer_pool_size = 8G  # 缓冲池大小（设为内存的 50-80%）
  innodb_buffer_pool_instances = 8  # 实例数（提高并发）
```

### 5. Change Buffer (写缓冲)

```
作用：优化非唯一二级索引的写入

问题：
  • 插入数据时，需要更新二级索引
  • 二级索引不是顺序的，可能导致随机 I/O
  • 如果索引页不在内存中，需要先读取（慢）

解决：Change Buffer
  1. 插入时，如果索引页不在内存，先写 Change Buffer
  2. 后台合并到索引页（Merge）

┌─────────────────────────────────────┐
│  INSERT INTO users (id, name)       │
│  VALUES (100, 'zhangsan');          │
│         ↓                           │
│  主键索引：直接写入 (聚簇索引)       │
│         ↓                           │
│  二级索引 (name)：                   │
│    索引页在内存 → 直接写入           │
│    索引页不在内存 → 写 Change Buffer │
└─────────────────────────────────────┘

合并时机：
  • 访问该索引页时
  • 后台线程定期合并
  • MySQL 关闭时

限制：
  • 只适用于非唯一索引
  • 唯一索引需要检查冲突，必须读取索引页

配置：
  innodb_change_buffering = all  # 缓冲所有操作
  innodb_change_buffer_max_size = 25  # 占 Buffer Pool 的 25%
```

## ⚡ MyISAM 存储引擎

MySQL 5.5 之前的默认引擎，现在很少使用。

### 特点
```
优点：
  ✓ 查询速度快（无事务开销）
  ✓ 表级锁，实现简单
  ✓ 支持全文索引

缺点：
  ❌ 不支持事务
  ❌ 不支持 MVCC
  ❌ 表级锁，并发写入差
  ❌ 崩溃后无法安全恢复

适用场景：
  • 只读或读多写少
  • 不需要事务
  • 日志、归档数据
```

### 存储结构
```
MyISAM 表由三个文件组成：

users.frm  # 表结构定义
users.MYD  # 数据文件 (MYData)
users.MYI  # 索引文件 (MYIndex)

索引组织：
  • 非聚簇索引
  • 索引和数据分离
  • 索引叶子节点存储数据文件的偏移量

┌─────────────────┐   ┌──────────────────┐
│  主键索引 (MYI)  │   │  数据文件 (MYD)   │
│                 │   │                  │
│   ┌───┐         │   │  ┌────────────┐  │
│   │ 5 │         │   │  │ 1, 张三, 25│  │
│  ┌┴─┬─┴┐        │   │  ├────────────┤  │
│ ┌┴┐┌┴┐┌┴┐       │   │  │ 2, 李四, 30│  │
│ │1││5││8│       │   │  ├────────────┤  │
│ └┬┘└┬┘└┬┘       │   │  │ 3, 王五, 28│  │
│  │  │  │        │   │  └────────────┘  │
│  ↓  ↓  ↓        │   │                  │
│ ptr ptr ptr ────┼───→  指向数据文件     │
└─────────────────┘   └──────────────────┘

特点：
  • 主键索引和二级索引结构相同
  • 都存储数据文件偏移量
  • 查询时通过偏移量直接定位数据
```

## 🌲 LSM 树存储引擎

用于写密集型场景，如 RocksDB、LevelDB、HBase。

### 核心思想
```
问题：B+ 树的写入性能
  • 随机写入
  • 页分裂
  • 写放大

LSM 树 (Log-Structured Merge Tree)：
  1. 写入时先写内存 (MemTable)
  2. 内存写满后刷到磁盘 (SSTable)
  3. 后台合并 SSTable (Compaction)

┌─────────────────────────────────────┐
│      LSM 树结构                      │
│                                     │
│  写入                                │
│   ↓                                 │
│  MemTable (内存)                     │
│   ↓ (写满)                           │
│  Immutable MemTable                 │
│   ↓ (刷盘)                           │
│  Level 0 SSTable                    │
│   ↓ (合并)                           │
│  Level 1 SSTable                    │
│   ↓                                 │
│  Level 2 SSTable                    │
│   ...                               │
└─────────────────────────────────────┘

优点：
  ✓ 顺序写入，写性能高
  ✓ 天然支持压缩
  ✓ 适合写多读少

缺点：
  ❌ 读取需要查多个 SSTable
  ❌ 合并过程消耗 CPU 和 I/O
  ❌ 写放大（一次写入可能多次合并）
```

### 读写流程

**写入流程**
```
1. 写 WAL (Write-Ahead Log)
   ↓
2. 写 MemTable (内存中的跳表)
   ↓
3. MemTable 写满 → 转为 Immutable MemTable
   ↓
4. 后台线程刷盘 → 生成 SSTable 文件
   ↓
5. 后台 Compaction 合并 SSTable

特点：
  • 写入全部是顺序的
  • 内存写入，延迟低
```

**读取流程**
```
1. 查 MemTable
   ↓
2. 查 Immutable MemTable
   ↓
3. 查 Level 0 SSTable (可能多个)
   ↓
4. 查 Level 1 SSTable
   ...

优化：
  • Bloom Filter: 快速判断 key 是否存在
  • Block Cache: 缓存热点数据
```

## 🔄 行存储 vs 列存储

### 行存储 (Row Store)
```
按行存储数据：

┌────┬──────┬─────┬────────┐
│ id │ name │ age │ salary │
├────┼──────┼─────┼────────┤
│ 1  │ 张三 │ 25  │ 8000   │  ← 行 1 在一起
│ 2  │ 李四 │ 30  │ 9000   │  ← 行 2 在一起
│ 3  │ 王五 │ 28  │ 7500   │  ← 行 3 在一起
└────┴──────┴─────┴────────┘

磁盘存储：[1,张三,25,8000][2,李四,30,9000][3,王五,28,7500]

优点：
  ✓ 插入、更新、删除快 (一次 I/O)
  ✓ 适合 OLTP (事务处理)

缺点：
  ❌ 分析查询慢 (需要扫描整行)
  ❌ 压缩率低

示例：
  SELECT * FROM users WHERE id = 1;  ← 快 (读一行)
  SELECT AVG(salary) FROM users;     ← 慢 (读所有行，但只需 salary 列)
```

### 列存储 (Column Store)
```
按列存储数据：

┌────┬──────┬─────┬────────┐
│ id │ name │ age │ salary │
├────┼──────┼─────┼────────┤
│ 1  │ 张三 │ 25  │ 8000   │
│ 2  │ 李四 │ 30  │ 9000   │
│ 3  │ 王五 │ 28  │ 7500   │
└────┴──────┴─────┴────────┘
  ↓     ↓     ↓      ↓
  │     │     │      │
  └─────┼─────┼──────┘
        ↓     ↓
磁盘存储：
  id: [1,2,3]
  name: [张三,李四,王五]
  age: [25,30,28]
  salary: [8000,9000,7500]

优点：
  ✓ 分析查询快 (只读需要的列)
  ✓ 压缩率高 (同类型数据在一起)
  ✓ 适合 OLAP (分析处理)

缺点：
  ❌ 插入、更新慢 (需要更新多个列文件)
  ❌ 不适合 OLTP

示例：
  SELECT AVG(salary) FROM users;     ← 快 (只读 salary 列)
  SELECT * FROM users WHERE id = 1;  ← 慢 (需要读多个列)

使用场景：
  • ClickHouse (OLAP)
  • Apache Parquet (大数据)
  • Apache Cassandra (部分列族)
```

## 🎯 最佳实践

### 1. 存储引擎选择
```sql
-- ✓ 大多数场景：InnoDB
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    user_id BIGINT,
    amount DECIMAL(10,2)
) ENGINE=InnoDB;

-- ✓ 只读归档表：MyISAM (或 Archive)
CREATE TABLE logs_2023 (
    id BIGINT,
    message TEXT
) ENGINE=Archive;

-- ✓ 临时表：Memory
CREATE TEMPORARY TABLE temp_data (
    id INT,
    value INT
) ENGINE=Memory;
```

### 2. 表设计原则
```sql
-- ✓ 使用自增主键
CREATE TABLE users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,  -- 顺序插入，避免页分裂
    username VARCHAR(50)
);

-- ❌ 使用 UUID 主键
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,  -- 随机插入，频繁页分裂
    username VARCHAR(50)
);

-- ✓ 独立表空间
innodb_file_per_table = ON

-- ✓ 合理设置 Buffer Pool
innodb_buffer_pool_size = 8G  -- 物理内存的 50-80%
```

### 3. 日志配置
```ini
# Redo Log
innodb_log_file_size = 512M
innodb_log_buffer_size = 16M
innodb_flush_log_at_trx_commit = 1  # 1=每次提交刷盘, 2=每秒刷盘

# Binlog
log_bin = ON
binlog_format = ROW
sync_binlog = 1  # 每次提交都刷盘

# Undo Log
innodb_undo_log_truncate = ON
innodb_max_undo_log_size = 1G
```

## 🔗 与其他概念的联系

### 与数据结构
- **B+ 树** - InnoDB 索引实现
- **跳表** - LSM 树的 MemTable
- **布隆过滤器** - LSM 树读取优化

参考：`fundamentals/data-structures/`

### 与操作系统
- **页** - 与操作系统的页概念类似
- **缓冲池** - 类似 OS 的页缓存
- **WAL** - 类似文件系统的日志

参考：`systems/operating-systems/`

### 与数据库其他概念
- [索引](./indexing-query-optimization.md) - 存储引擎实现索引
- [事务](./transactions-concurrency.md) - 通过日志实现 ACID

## 📚 扩展阅读

### 深入理解
- InnoDB 源码分析
- RocksDB 设计与实现
- 存储引擎性能对比

### 前沿技术
- NVMe SSD 优化
- PMEM (持久内存) 存储引擎
- 分布式存储引擎

---

**理解存储引擎，你就掌握了数据库性能优化的关键！** 💾
