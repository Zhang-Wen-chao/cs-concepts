# 数据库系统 (Database Systems)

> 高效存储、管理和检索数据的核心技术

## 📚 本模块内容

数据库系统是现代应用的基石，理解数据库能帮助我们：
- 设计高效的数据模型
- 编写高性能的查询
- 保证数据一致性和可靠性
- 处理大规模并发访问

## 🏗️ 核心概念

### 1. [数据库基础概念](./database-fundamentals.md)
> 什么是数据库？为什么需要数据库？

**核心内容：**
- 数据库 vs 文件系统
- 数据库管理系统 (DBMS)
- 数据模型：层次、网状、关系、文档、图
- 三级模式结构
- 数据独立性

**关键概念：**
- ACID 特性
- 数据完整性
- 数据冗余
- 数据抽象

### 2. [关系模型与 SQL](./relational-model-sql.md)
> 关系型数据库的理论基础和查询语言

**核心内容：**
- 关系模型基础（表、行、列、键）
- SQL 基础（SELECT、INSERT、UPDATE、DELETE）
- 连接操作（INNER、LEFT、RIGHT、FULL）
- 子查询与嵌套查询
- 聚合函数与分组
- 视图、存储过程、触发器

**关键概念：**
- 主键与外键
- 范式理论（1NF、2NF、3NF、BCNF）
- 关系代数
- SQL 执行顺序

### 3. [索引与查询优化](./indexing-query-optimization.md)
> 如何让查询跑得更快

**核心内容：**
- 索引原理
- B+ 树索引
- 哈希索引
- 覆盖索引与联合索引
- 查询执行计划
- 查询优化器
- 统计信息

**关键概念：**
- 索引选择性
- 回表 (Lookup)
- 索引下推
- EXPLAIN 分析
- 代价模型

### 4. [事务与并发控制](./transactions-concurrency.md)
> 保证数据一致性的关键机制

**核心内容：**
- 事务的 ACID 特性
- 隔离级别（Read Uncommitted、Read Committed、Repeatable Read、Serializable）
- 并发异常（脏读、不可重复读、幻读）
- 锁机制（共享锁、排他锁、意向锁）
- 多版本并发控制 (MVCC)
- 死锁检测与处理

**关键概念：**
- 两阶段锁协议
- 乐观锁 vs 悲观锁
- 时间戳排序
- 快照隔离

### 5. [存储引擎](./storage-engines.md)
> 数据在磁盘上如何组织和存储

**核心内容：**
- 存储结构（页、区、段）
- InnoDB vs MyISAM
- 行存储 vs 列存储
- LSM 树（Log-Structured Merge Tree）
- 缓冲池管理
- 日志系统（Redo Log、Undo Log、Binary Log）

**关键概念：**
- 聚簇索引 vs 非聚簇索引
- 写放大问题
- WAL (Write-Ahead Logging)
- 崩溃恢复

## 🔗 概念关联图

```
数据库系统
│
├─ 数据模型层
│  ├─ 关系模型 → SQL 查询语言
│  ├─ 范式理论 → 数据库设计
│  └─ 完整性约束 → 数据一致性
│
├─ 存储层
│  ├─ 存储引擎 → 数据组织方式
│  ├─ 索引结构 → 查询加速
│  └─ 缓冲管理 → 内存优化
│
├─ 执行层
│  ├─ 查询优化器 → 生成执行计划
│  ├─ 执行引擎 → 执行查询
│  └─ 统计信息 → 优化决策
│
└─ 事务层
   ├─ 并发控制 → 隔离性
   ├─ 日志系统 → 持久性与恢复
   └─ 锁管理 → 一致性
```

## 🎯 学习路径

### 初学者路径
1. **数据库基础** - 理解为什么需要数据库
2. **关系模型与 SQL** - 学会查询和操作数据
3. **索引基础** - 理解如何加速查询
4. **事务基础** - 理解 ACID 特性

### 进阶路径
1. **查询优化** - 分析和优化慢查询
2. **并发控制深入** - MVCC、死锁处理
3. **存储引擎原理** - B+ 树、LSM 树
4. **分布式数据库** - 分片、复制、一致性

## 💡 为什么要学习数据库？

### 日常开发必备
- 设计合理的表结构
- 编写高效的 SQL 查询
- 处理数据一致性问题
- 优化数据库性能

### 理解系统瓶颈
```
大多数 Web 应用的性能瓶颈在数据库：
  • CPU 计算: 很快
  • 网络传输: 较快
  • 数据库查询: 慢！

理解数据库 → 找到性能瓶颈 → 有效优化
```

### 实际应用案例

**案例 1：电商系统**
```sql
-- 问题：查询用户订单很慢
SELECT * FROM orders WHERE user_id = 123;

-- 原因：没有索引，全表扫描
-- 解决：添加索引
CREATE INDEX idx_user_id ON orders(user_id);

性能提升：从 5 秒到 0.01 秒
```

**案例 2：库存扣减**
```sql
-- 问题：高并发下库存超卖
UPDATE inventory SET stock = stock - 1 WHERE product_id = 456;

-- 原因：没有事务和锁
-- 解决：使用事务和行锁
START TRANSACTION;
SELECT stock FROM inventory WHERE product_id = 456 FOR UPDATE;
-- 检查库存是否足够
UPDATE inventory SET stock = stock - 1 WHERE product_id = 456;
COMMIT;
```

**案例 3：分页查询**
```sql
-- 问题：深分页很慢
SELECT * FROM posts ORDER BY create_time DESC LIMIT 1000000, 20;

-- 原因：需要扫描前 1000000 行
-- 解决：使用游标或 id 范围
SELECT * FROM posts
WHERE id > last_seen_id
ORDER BY id DESC
LIMIT 20;
```

## 🔧 与其他概念的联系

### 与操作系统
- **文件系统** - 数据库底层使用文件存储
- **缓冲管理** - 类似 OS 的页缓存
- **锁机制** - 类似进程同步

参考：`systems/operating-systems/`

### 与数据结构
- **B+ 树** - 索引的核心数据结构
- **哈希表** - 哈希索引
- **跳表** - Redis 的有序集合

参考：`fundamentals/data-structures/`

### 与算法
- **查询优化** - 图算法（连接优化）
- **排序算法** - ORDER BY 实现
- **动态规划** - 查询计划选择

参考：`fundamentals/algorithms/`

### 与网络
- **客户端-服务器架构** - 数据库连接
- **分布式系统** - 主从复制、分片

参考：`systems/networks/`

## 📖 推荐学习资源

### 经典教材
- *Database System Concepts* (Silberschatz) - 数据库圣经
- *Designing Data-Intensive Applications* (Martin Kleppmann) - 现代视角
- *High Performance MySQL* - MySQL 优化

### 实践项目
- 设计并实现简单的数据库引擎
- 分析真实应用的慢查询并优化
- 阅读 MySQL/PostgreSQL 源码

### 在线资源
- MySQL 官方文档
- PostgreSQL 文档
- CMU 数据库课程 (15-445/645)

## 🎓 学习建议

1. **先理论再实践**
   - 理解概念原理
   - 然后动手写 SQL
   - 分析执行计划

2. **关注性能**
   - 学会使用 EXPLAIN
   - 理解索引原理
   - 监控慢查询

3. **重视事务**
   - 理解隔离级别
   - 避免并发问题
   - 处理死锁

4. **实际应用导向**
   - 解决真实问题
   - 优化实际查询
   - 设计合理的表结构

## 🔍 数据库选型

### 关系型数据库

#### MySQL vs PostgreSQL 深度对比

**架构设计：**
```
MySQL:
  • 多存储引擎架构（InnoDB、MyISAM 等）
  • 可插拔式存储引擎
  • 存储引擎层和服务器层分离

PostgreSQL:
  • 单一集成架构
  • 功能更统一，扩展性强
  • 插件式扩展（如 PostGIS）
```

**MVCC 实现：**
```
MySQL (InnoDB):
  • 通过 Undo Log 实现多版本
  • 旧版本存储在回滚段
  • 需要定期清理 Undo Log

PostgreSQL:
  • 通过多版本元组实现
  • 新旧版本都存储在表中
  • VACUUM 清理旧版本
```

**事务与并发：**
```
MySQL:
  • 默认隔离级别：Repeatable Read
  • 通过 Next-Key Lock 解决幻读
  • 死锁检测：超时回滚

PostgreSQL:
  • 默认隔离级别：Read Committed
  • Serializable Snapshot Isolation (SSI)
  • 更严格的 ACID 保证
```

**复制方式：**
```
MySQL:
  • 基于 Binlog（二进制日志）
  • 三种格式：Statement/Row/Mixed
  • 主从复制配置简单

PostgreSQL:
  • 基于 WAL（Write-Ahead Log）
  • 物理复制 + 逻辑复制
  • 流复制支持同步/异步
```

**高级特性：**
```
PostgreSQL 优势：
  ✅ 窗口函数支持更完善
  ✅ CTE（公共表表达式）和递归查询
  ✅ 原生 JSON/JSONB 支持
  ✅ 数组、范围类型
  ✅ 全文搜索（内置）
  ✅ 表继承、分区表
  ✅ 更强的查询优化器

MySQL 优势：
  ✅ 生态更丰富（工具、社区）
  ✅ 学习曲线平缓
  ✅ 主从复制更简单
  ✅ 性能调优资料多
```

**使用场景：**
```
选择 MySQL：
  • Web 应用（LAMP/LNMP）
  • 读多写少
  • 需要简单的主从复制
  • 团队熟悉 MySQL

选择 PostgreSQL：
  • 复杂查询和分析
  • 需要严格 ACID
  • 使用 JSON 文档
  • GIS 应用（PostGIS）
  • 数据仓库
```

#### 其他关系型数据库

**SQLite:**
```
  ✅ 嵌入式，零配置
  ✅ 轻量级
  ❌ 不支持高并发

  适用场景：
  • 移动应用
  • 桌面软件
  • 小型网站
```

### NoSQL 数据库
```
Redis:
  • 键值存储，超快
  • 缓存、会话存储

MongoDB:
  • 文档数据库
  • Schema 灵活

Cassandra:
  • 分布式列存储
  • 高可用、可扩展
```

---

**开始探索数据库的世界，掌握数据管理的核心技术！** 🗄️
