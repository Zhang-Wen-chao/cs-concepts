# 数据库基础概念 (Database Fundamentals)

> 什么是数据库？为什么需要数据库？

## 🎯 核心概念

**数据库 = 有组织地存储和管理数据的系统**
**DBMS = 管理数据库的软件**，提供数据抽象、独立性、高效访问、完整性、并发控制、恢复和安全性。

---

## ⚡ 数据库 vs 文件系统

| 维度 | 文件系统 | 数据库 |
|------|---------|--------|
| 数据冗余 | ❌ 多文件重复存储 | ✅ 统一管理 |
| 访问方式 | 手写解析代码 | SQL 查询 |
| 并发控制 | 应用自建锁 | 内置事务/锁 |
| 完整性 | 应用自检 | 约束/触发器 |
| 恢复能力 | 易丢失 | 日志+回滚 |

---

## 🏛️ 三级模式架构

```
          外模式 (External Schema)
        用户看到的数据视图 (View 1/2/3)
                ↓ 外模式/模式映射
         概念模式 (Conceptual Schema)
        数据库的逻辑结构 (表/约束/关系)
                ↓ 模式/内模式映射
         内模式 (Internal Schema)
        物理存储 (文件/索引/压缩/加密)
                ↓
               磁盘
```

**数据独立性**：修改内模式不影响概念/外模式（物理独立性）；修改概念模式可通过视图屏蔽影响（逻辑独立性）。

---

## 🗺️ 数据模型演进

```
层次模型 (树) → 网状模型 (图) → 关系模型 (表) ← 最成功
                                            ↓
                                   对象模型 / NoSQL 模型
```

### 关系模型 (Relational Model)
- 数据组织为**表**（关系），行=元组，列=属性
- 通过**外键**建立表间关系
- 数学基础：关系代数
- SQL 作为操作语言

### NoSQL 模型
| 类型 | 代表 | 场景 |
|------|------|------|
| 键值 | Redis | 缓存、会话 |
| 文档 | MongoDB | 灵活 Schema |
| 列族 | Cassandra | 宽表、时序 |
| 图 | Neo4j | 社交、推荐 |

---

## 🧩 DBMS 组件

```
用户/应用程序 → 查询处理器 (Parser → Optimizer → Executor)
                                  ↓
              存储管理器 (事务/缓冲/文件/索引管理器)
                                  ↓
              磁盘存储 (数据/索引/日志文件)
```

**查询处理流程**：SQL → 解析树 → 优化 → 执行计划 → 结果集

---

## ✅ ACID 四大特性

| 特性 | 含义 | 实现 |
|------|------|------|
| 原子性 (Atomicity) | 事务全做或全不做 | Undo Log |
| 一致性 (Consistency) | 事务前后数据一致 | 约束+业务规则 |
| 隔离性 (Isolation) | 并发互不干扰 | 锁/MVCC |
| 持久性 (Durability) | 提交后永久保存 | Redo Log + WAL |

---

## 🔐 数据完整性约束

- **实体完整性**：主键唯一且非 NULL
- **参照完整性**：外键引用已存在主键
- **用户定义完整性**：CHECK、UNIQUE 等

---

## 🗣️ 数据库语言分类

| 分类 | 命令 | 作用 |
|------|------|------|
| DDL | CREATE/ALTER/DROP | 定义结构 |
| DML | SELECT/INSERT/UPDATE/DELETE | 操作数据 |
| DCL | GRANT/REVOKE | 权限控制 |
| TCL | COMMIT/ROLLBACK/SAVEPOINT | 事务管理 |

---

## 🔗 与其他概念的联系

- **OS (文件系统/缓冲管理/进程)** → `systems/operating-systems/`
- **数据结构 (B+树/哈希表/队列)** → `fundamentals/data-structures/`
- **网络 (客户端-服务器/协议)** → `systems/networks/`

## 📚 深入学习

- *Database System Concepts* - Silberschatz（教材）
- CMU 15-445: Database Systems（课程）

### 下一步
- [关系模型与 SQL](./relational-model-sql.md)
- [索引与查询优化](./indexing-query-optimization.md)
- [事务与并发控制](./transactions-concurrency.md)

> 详细内容与代码示例详见 [practices/systems/databases/](../../practices/systems/databases/)
