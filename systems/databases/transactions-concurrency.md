# 事务与并发控制 (Transactions and Concurrency Control)

> 保证数据一致性和正确性的核心机制

## 🎯 核心思想

- **事务**：多个操作打包为**原子单元**，要么全做要么全不做
- **并发控制**：多事务**安全地同时执行**，互不干扰

---

## ✅ ACID 特性

| 特性 | 含义 | 实现机制 |
|------|------|----------|
| 原子性 (A) | 全做或全不做 | Undo Log（记录反向操作） |
| 一致性 (C) | 事务前后数据一致 | 约束 + 业务规则 |
| 隔离性 (I) | 并发互不干扰 | 锁 / MVCC |
| 持久性 (D) | 提交后永保存 | Redo Log + WAL |

---

## 🔒 并发异常（三类问题）

```
脏读 (Dirty Read)
  T1 写未提交 → T2 读到 → T1 回滚 → T2 读到脏数据

不可重复读 (Non-Repeatable Read)
  T1 读 500 → T2 修改提交 → T1 再读 1000（值变了）

幻读 (Phantom Read)
  T1 查 10 条 → T2 插入提交 → T1 再查 11 条（数量变了）
```

---

## 🎚️ 隔离级别

| 级别 | 脏读 | 不可重复读 | 幻读 | 实现方式 |
|------|------|-----------|------|----------|
| Read Uncommitted | ✅ | ✅ | ✅ | 读不加锁，性能最高 |
| Read Committed | ❌ | ✅ | ✅ | 读加短锁，读完释放 |
| Repeatable Read | ❌ | ❌ | ⚠️* | 读加长锁+MVCC快照 |
| Serializable | ❌ | ❌ | ❌ | 范围锁 / 串行执行 |

*MySQL InnoDB RR 通过 Next-Key Lock 解决幻读

**隔离级别越高 → 锁越多/快照越久 → 并发吞吐量越低**

---

## 🔐 锁机制

### 锁类型
```
共享锁 (S) : 可共同持有，阻止写 → SELECT ... LOCK IN SHARE MODE
排他锁 (X) : 独占，阻止其他读写 → SELECT ... FOR UPDATE / UPDATE / DELETE
```

### 锁粒度
```
行锁 (InnoDB) → 页锁 → 表锁 (MyISAM)  (越细粒度并发越高)

意向锁 (IS/IX): 表级标记，提高加表锁效率
```

### 锁兼容矩阵
```
     | IS | IX | S  | X
─────┼────┼────┼────┼────
 IS  | ✓  | ✓  | ✓  | ❌
 IX  | ✓  | ✓  | ❌ | ❌
 S   | ✓  | ❌ | ✓  | ❌
 X   | ❌ | ❌ | ❌ | ❌
```

### 两阶段锁 (2PL)
```
加锁阶段(只加不释) → 解锁阶段(只释不加)
保证可串行化，但锁持有时间长 → 死锁风险
```

---

## 👻 MVCC (多版本并发控制)

**核心思想**：读不加锁，读写不冲突

```
每行隐藏列: DB_TRX_ID(事务ID) | DB_ROLL_PTR(指向Undo旧版本)

当前行 (txn=20) → Undo: v2 (txn=15) → Undo: v1 (txn=10)
                   ↑ 通过 DB_ROLL_PTR 链回溯老版本
```

**Read View**（一致性视图）
```
RC: 每次查询创建新 Read View
RR: 事务开始时创建一次 Read View，之后复用

可见性规则：
  txn_id < min(m_ids)  → 可见
  txn_id in m_ids      → 不可见（未提交）
  txn_id = 自己        → 可见
```

---

## 💀 死锁

```
T1: LOCK A → wait B    T2: LOCK B → wait A   →  死锁!
```

**数据库处理**：维护等待图，检测到环→选牺牲者回滚

**预防**：
1. 按固定顺序加锁（如先锁小 id）
2. 短事务，避免事务内做网络/复杂计算
3. 乐观锁（版本号）+ 重试

---

## 🎯 最佳实践

- **事务尽可能短**，不在事务内做网络请求/复杂计算
- **`SELECT ... FOR UPDATE`** 用于先查后改（库存扣减）
- **`SELECT ... LOCK IN SHARE MODE`** 用于读一致性数据
- **死锁捕获 + 重试**：程序端处理死锁异常

---

## 🔗 与其他概念的联系

- **OS (锁/死锁/写时复制)** → `systems/operating-systems/process-synchronization.md`
- **关系模型** → [relational-model-sql.md](./relational-model-sql.md)
- **索引 (锁粒度和索引相关)** → [indexing-query-optimization.md](./indexing-query-optimization.md)
- **存储引擎 (日志实现ACID)** → [storage-engines.md](./storage-engines.md)

## 📚 扩展阅读

- "A Critique of ANSI SQL Isolation Levels" (1995) — 经典论文
- 乐观并发控制 (OCC)、快照隔离 (SI)、分布式事务 (2PC/Saga)

> 详细事务示例与并发测试详见 [practices/systems/databases/](../../practices/systems/databases/)
