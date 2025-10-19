# 事务与并发控制 (Transactions and Concurrency Control)

> 保证数据一致性和正确性的核心机制

## 🎯 核心思想

在多用户同时访问数据库的环境下，如何保证数据的一致性和正确性？答案是**事务和并发控制**。

**一句话理解：**
- 事务：把多个操作打包成一个**原子单元**，要么全做，要么全不做
- 并发控制：让多个事务**安全地同时执行**，不会互相干扰

## 📖 事务基础

### 1. 什么是事务？

```
经典例子：银行转账

张三给李四转账 100 元：
  1. 从张三账户扣 100
  2. 给李四账户加 100

问题：如果执行到第 1 步时系统崩溃了怎么办？
  → 张三的钱扣了，李四没收到
  → 数据不一致！

解决：使用事务
START TRANSACTION;
  UPDATE account SET balance = balance - 100 WHERE id = 'zhangsan';
  UPDATE account SET balance = balance + 100 WHERE id = 'lisi';
COMMIT;

要么两个操作都成功，要么都不做
```

### 2. ACID 特性

#### A - 原子性 (Atomicity)
```
事务是不可分割的最小单位

START TRANSACTION;
  INSERT INTO orders (id, user_id) VALUES (1, 100);
  INSERT INTO order_items (order_id, product_id) VALUES (1, 200);
COMMIT;

结果：
  ✓ 两条 INSERT 都成功 → 提交
  ✓ 任何一条失败 → 全部回滚

不会出现：
  ❌ orders 插入成功，order_items 插入失败
```

**实现机制：Undo Log**
```
数据库记录每个操作的反向操作

操作：UPDATE account SET balance = 1000 WHERE id = 1
Undo Log: UPDATE account SET balance = 900 WHERE id = 1

回滚时：执行 Undo Log 中的反向操作
```

#### C - 一致性 (Consistency)
```
事务执行前后，数据库从一个一致状态变为另一个一致状态

示例：转账前后总金额不变
转账前：张三 500 + 李四 300 = 800
转账后：张三 400 + 李四 400 = 800  ✓

不一致的状态：
转账后：张三 400 + 李四 300 = 700  ❌ (丢了 100)

一致性约束：
  • 主键唯一
  • 外键引用有效
  • 余额不能为负
  • 业务规则（如总金额守恒）
```

#### I - 隔离性 (Isolation)
```
多个事务并发执行时，互不干扰

场景：两个人同时给张三转账

事务 A：给张三转账 100
事务 B：给张三转账 200

如果没有隔离：
  A 读取余额：500
  B 读取余额：500
  A 写入：500 + 100 = 600
  B 写入：500 + 200 = 700
  结果：最终余额 700（丢失了 A 的更新）

有隔离：
  A 和 B 顺序执行
  最终余额：500 + 100 + 200 = 800  ✓
```

#### D - 持久性 (Durability)
```
事务一旦提交，修改就是永久的，即使系统崩溃也不会丢失

COMMIT 后：
  ✓ 数据写入磁盘
  ✓ 系统崩溃后能恢复
  ✓ 对用户承诺已生效

实现机制：Redo Log
  • 先写日志，再写数据
  • 崩溃后根据日志恢复
```

### 3. 事务的生命周期

```
┌────────────────────────────────────────────────┐
│                                                │
│  START TRANSACTION (BEGIN)                     │
│         ↓                                      │
│  执行 SQL 语句 (SELECT, INSERT, UPDATE, DELETE) │
│         ↓                                      │
│  ┌─────────────┐         ┌──────────┐         │
│  │   COMMIT    │   or    │ ROLLBACK │         │
│  │  (提交事务)  │         │ (回滚事务) │         │
│  └─────────────┘         └──────────┘         │
│         ↓                       ↓              │
│  事务成功完成              事务撤销              │
│  修改永久保存              恢复到初始状态         │
│                                                │
└────────────────────────────────────────────────┘

示例：
START TRANSACTION;
  UPDATE account SET balance = balance - 100 WHERE id = 1;
  -- 检查余额是否足够
  SELECT balance FROM account WHERE id = 1;
  IF (balance < 0) THEN
    ROLLBACK;  -- 余额不足，回滚
  ELSE
    COMMIT;    -- 余额足够，提交
  END IF;
```

## 🔒 并发问题

### 1. 三种并发异常

#### 脏读 (Dirty Read)
```
读取到其他事务未提交的数据

时间线：
T1: START TRANSACTION;
T1: UPDATE account SET balance = 1000 WHERE id = 1;
                                            T2: START TRANSACTION;
                                            T2: SELECT balance FROM account WHERE id = 1;
                                            T2: 读到 1000 (T1 未提交的数据)
T1: ROLLBACK;  -- T1 回滚了！

结果：T2 读到了"脏数据" (1000)，但 T1 最终回滚了

问题：
  • T2 基于错误数据做决策
  • T1 的中间状态被 T2 看到了
```

#### 不可重复读 (Non-Repeatable Read)
```
同一事务中，两次读取同一数据，结果不同

时间线：
T1: START TRANSACTION;
T1: SELECT balance FROM account WHERE id = 1;  -- 读到 500
                                            T2: START TRANSACTION;
                                            T2: UPDATE account SET balance = 1000 WHERE id = 1;
                                            T2: COMMIT;
T1: SELECT balance FROM account WHERE id = 1;  -- 读到 1000
T1: COMMIT;

结果：T1 两次读取同一数据，结果不同 (500 → 1000)

问题：
  • T1 内部数据不一致
  • 无法保证事务内的读取稳定性
```

#### 幻读 (Phantom Read)
```
同一事务中，两次查询返回的记录数不同

时间线：
T1: START TRANSACTION;
T1: SELECT COUNT(*) FROM orders WHERE user_id = 1;  -- 查到 10 条
                                                   T2: START TRANSACTION;
                                                   T2: INSERT INTO orders VALUES (11, 1, ...);
                                                   T2: COMMIT;
T1: SELECT COUNT(*) FROM orders WHERE user_id = 1;  -- 查到 11 条
T1: COMMIT;

结果：T1 两次查询记录数不同 (10 → 11)

问题：
  • 出现了"幻影"记录
  • 范围查询不稳定

与不可重复读的区别：
  • 不可重复读：同一条记录的值变了
  • 幻读：记录数变了
```

### 2. 四种隔离级别

```
隔离级别          脏读    不可重复读    幻读      并发性能
─────────────────────────────────────────────────────
Read Uncommitted   ✓       ✓          ✓       最高
Read Committed     ❌      ✓          ✓       较高
Repeatable Read    ❌      ❌         ✓       较低
Serializable       ❌      ❌         ❌      最低

✓ = 会出现
❌ = 不会出现

性能说明：
• 隔离级别越高 → 加的锁越多/保留快照越久 → 并发性能越低
• "性能"指的是并发吞吐量（单位时间内完成的事务数）
• Read Uncommitted：几乎不加锁，但数据不可靠
• Serializable：最严格的锁/检查，吞吐量最低
```

**隔离级别与性能的关系：**

```
为什么隔离级别影响性能？

1. 锁的数量和范围
   Read Uncommitted: 读不加锁 → 无等待
   Read Committed:   读加短锁 → 读完立即释放
   Repeatable Read:  读加长锁 → 事务结束才释放
   Serializable:     范围锁   → 阻止其他事务访问范围

2. 冲突检测开销
   低隔离级别：不检测或检测少 → 开销小
   高隔离级别：检测更多冲突   → 开销大

3. 并发度
   锁范围越大 → 冲突越多 → 等待越多 → 并发度降低

实际影响：
  Read Uncommitted vs Serializable
  在高并发下，吞吐量可能相差 10-100 倍
```

#### Read Uncommitted (读未提交)
```sql
-- 设置隔离级别
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

特点：
  • 可以读取其他事务未提交的数据
  • 三种异常都会出现
  • 性能最高，但几乎不使用

示例：
T1: UPDATE account SET balance = 1000 WHERE id = 1;  -- 未提交
T2: SELECT balance FROM account WHERE id = 1;        -- 读到 1000 (脏读)

实现机制：
• 写操作加排他锁（X锁），事务结束释放
• 读操作不加锁 ← 性能最高的原因
• 允许读取未提交数据

性能特点：
✅ 读操作无锁，无等待
✅ 并发度最高
❌ 数据不可靠，几乎不使用
```

#### Read Committed (读已提交)
```sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

特点：
  • 只能读取已提交的数据
  • 避免脏读
  • 但会出现不可重复读和幻读

实现：每次查询都读取最新的已提交数据

示例：
T1: SELECT balance FROM account WHERE id = 1;  -- 500
T2: UPDATE account SET balance = 1000 WHERE id = 1;
T2: COMMIT;
T1: SELECT balance FROM account WHERE id = 1;  -- 1000 (不可重复读)

PostgreSQL、Oracle、SQL Server 默认级别

实现机制：
• 写操作加排他锁，事务结束释放
• 读操作加共享锁，读完立即释放 ← 性能较好
• 只读取已提交的数据版本

性能特点：
✅ 读锁持有时间短，减少等待
✅ 适合大多数应用
❌ 可能出现不可重复读
```

#### Repeatable Read (可重复读)
```sql
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

特点：
  • 同一事务内多次读取同一数据，结果相同
  • 避免脏读和不可重复读
  • 但会出现幻读（MySQL InnoDB 通过 MVCC 和 Next-Key Lock 解决）

实现：事务开始时创建快照，读取快照数据

示例：
T1: SELECT balance FROM account WHERE id = 1;  -- 500
T2: UPDATE account SET balance = 1000 WHERE id = 1;
T2: COMMIT;
T1: SELECT balance FROM account WHERE id = 1;  -- 仍然是 500 (可重复读)

MySQL InnoDB 默认级别

实现机制：
• 写操作加排他锁，事务结束释放
• 读操作加共享锁，事务结束释放 ← 锁持有时间长
• 使用 MVCC 读取事务开始时的快照

性能特点：
⚠️  读锁持有时间长，可能阻塞写操作
⚠️  快照保留时间长，占用更多内存
✅ 提供了较好的隔离性
```

#### Serializable (串行化)
```sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

特点：
  • 最高隔离级别
  • 完全避免三种异常
  • 事务串行执行（或看起来串行）
  • 性能最差

实现：对所有读取的数据加锁

示例：
T1: SELECT * FROM account;  -- 对所有行加锁
T2: UPDATE account SET balance = 1000;  -- 等待 T1 释放锁

实现机制：
• 使用范围锁（Range Lock）
• 读操作也会阻止其他事务的写入
• 或使用乐观并发控制（冲突检测 + 回滚）

性能特点：
❌ 并发度最低，大量事务等待
❌ 可能导致死锁概率增加
❌ 吞吐量显著下降
✅ 数据一致性最强

很少使用，只在对一致性要求极高时使用
```

### 3. 隔离级别选择

```
场景 1：电商订单查询
  • 读多写少
  • 对一致性要求不高
  • 选择：Read Committed

场景 2：银行转账
  • 对一致性要求高
  • 不能出现不可重复读
  • 选择：Repeatable Read

场景 3：报表统计
  • 需要完全一致的快照
  • 可以接受性能损失
  • 选择：Serializable

实际应用：
  • 大多数应用：Read Committed 或 Repeatable Read
  • Read Uncommitted：几乎不用
  • Serializable：极少使用
```

## 🔐 并发控制机制

### 1. 锁机制

#### 锁的类型

**共享锁 (Shared Lock / S Lock / 读锁)**
```sql
-- 加共享锁
SELECT * FROM account WHERE id = 1 LOCK IN SHARE MODE;

特点：
  • 多个事务可以同时持有共享锁
  • 持有共享锁时，不能修改数据
  • 其他事务不能加排他锁

用途：读取数据时防止被修改

示例：
T1: SELECT * FROM account WHERE id = 1 LOCK IN SHARE MODE;  ✓
T2: SELECT * FROM account WHERE id = 1 LOCK IN SHARE MODE;  ✓ (可以同时读)
T3: UPDATE account SET balance = 1000 WHERE id = 1;         ❌ (等待锁释放)
```

**排他锁 (Exclusive Lock / X Lock / 写锁)**
```sql
-- 加排他锁
SELECT * FROM account WHERE id = 1 FOR UPDATE;

特点：
  • 只有一个事务可以持有排他锁
  • 其他事务不能加任何锁
  • 用于修改数据

示例：
T1: SELECT * FROM account WHERE id = 1 FOR UPDATE;  ✓
T2: SELECT * FROM account WHERE id = 1;             ✓ (普通读取不加锁，MVCC)
T2: SELECT * FROM account WHERE id = 1 FOR UPDATE;  ❌ (等待锁释放)
T3: UPDATE account SET balance = 1000 WHERE id = 1; ❌ (等待锁释放)
```

#### 锁的粒度

```
行锁 (Row Lock)：
  • 锁定单行数据
  • 并发度最高
  • InnoDB 默认使用

表锁 (Table Lock)：
  • 锁定整张表
  • 并发度最低
  • MyISAM 使用

页锁 (Page Lock)：
  • 锁定一页数据（介于行锁和表锁之间）
  • BerkeleyDB 使用

意向锁 (Intention Lock)：
  • 表级锁，表示事务想要对表中的行加锁
  • IS (意向共享锁) 和 IX (意向排他锁)
  • 用于提高加表锁的效率
```

**锁兼容矩阵**
```
       │ IS  │ IX  │ S   │ X
───────┼─────┼─────┼─────┼─────
   IS  │  ✓  │  ✓  │  ✓  │  ❌
   IX  │  ✓  │  ✓  │  ❌ │  ❌
   S   │  ✓  │  ❌ │  ✓  │  ❌
   X   │  ❌ │  ❌ │  ❌ │  ❌

✓ = 兼容
❌ = 不兼容（需要等待）
```

#### 两阶段锁协议 (2PL)
```
事务分为两个阶段：
  1. 加锁阶段：只能加锁，不能释放锁
  2. 解锁阶段：只能释放锁，不能加锁

时间线：
START TRANSACTION;
  加锁 → 加锁 → 加锁 → 操作数据 → 释放锁 → 释放锁
  └─── 加锁阶段 ───┘              └─ 解锁阶段 ─┘
COMMIT;

作用：
  • 保证可串行化 (Serializability)
  • 避免并发异常

问题：
  • 持有锁时间长
  • 可能导致死锁
```

### 2. MVCC (多版本并发控制)

```
核心思想：读不加锁，读写不冲突

原理：
  • 为每行数据维护多个版本
  • 每个版本有时间戳（事务 ID）
  • 读取时选择合适的版本

┌─────────────────────────────────────┐
│ account 表 (id=1)                   │
├─────────────────────────────────────┤
│ 版本 1: balance=500,  txn_id=10     │
│ 版本 2: balance=600,  txn_id=15     │
│ 版本 3: balance=1000, txn_id=20     │
└─────────────────────────────────────┘

T1 (txn_id=12) 读取：
  • 看到版本 1 (balance=500)
  • 版本 2 和 3 是未来版本，看不到

T2 (txn_id=18) 读取：
  • 看到版本 2 (balance=600)
  • 版本 3 是未来版本，看不到

T3 (txn_id=25) 读取：
  • 看到版本 3 (balance=1000)
  • 最新版本
```

**InnoDB 的 MVCC 实现**
```
每行记录的隐藏列：
  • DB_TRX_ID: 创建或最后修改该行的事务 ID
  • DB_ROLL_PTR: 指向 Undo Log 中的旧版本
  • DB_ROW_ID: 隐藏主键（如果表没有主键）

account 表的实际存储：
┌────┬─────────┬────────────┬─────────────┬──────────┐
│ id │ balance │ DB_TRX_ID  │ DB_ROLL_PTR │ DB_ROW_ID│
├────┼─────────┼────────────┼─────────────┼──────────┤
│ 1  │ 1000    │ 20         │ ptr → v2    │ ...      │
└────┴─────────┴────────────┴─────────────┴──────────┘
                                  ↓
                        Undo Log: balance=600, txn=15
                                  ↓
                        Undo Log: balance=500, txn=10

读取时：
  1. 读取当前行
  2. 检查 DB_TRX_ID 是否可见
  3. 如果不可见，通过 DB_ROLL_PTR 找旧版本
  4. 重复步骤 2-3 直到找到可见版本
```

**Read View (一致性视图)**
```
事务开始时创建 Read View，记录：
  • m_ids: 当前活跃的事务 ID 列表
  • min_trx_id: 最小的活跃事务 ID
  • max_trx_id: 下一个将被分配的事务 ID
  • creator_trx_id: 创建该 Read View 的事务 ID

可见性判断：
  IF (DB_TRX_ID < min_trx_id):
    可见 (在 Read View 创建前就提交了)
  ELSE IF (DB_TRX_ID >= max_trx_id):
    不可见 (在 Read View 创建后才开始)
  ELSE IF (DB_TRX_ID in m_ids):
    不可见 (还在活跃中，未提交)
  ELSE IF (DB_TRX_ID == creator_trx_id):
    可见 (自己的修改)
  ELSE:
    可见 (在 Read View 创建前提交)

不同隔离级别的 Read View：
  • Read Committed: 每次查询创建新的 Read View
  • Repeatable Read: 事务开始时创建一次 Read View
```

### 3. 死锁

#### 什么是死锁？
```
两个或多个事务相互等待对方释放锁

经典示例：
T1: LOCK A → wait for B
T2: LOCK B → wait for A
结果：互相等待，永远无法继续

具体场景：
T1: START TRANSACTION;
T1: UPDATE account SET balance = 1000 WHERE id = 1;  -- 锁住 id=1
                                                    T2: START TRANSACTION;
                                                    T2: UPDATE account SET balance = 2000 WHERE id = 2;  -- 锁住 id=2
T1: UPDATE account SET balance = 3000 WHERE id = 2;  -- 等待 T2 释放 id=2
                                                    T2: UPDATE account SET balance = 4000 WHERE id = 1;  -- 等待 T1 释放 id=1

死锁！
```

#### 死锁检测
```
数据库维护等待图 (Wait-For Graph)：

T1 → T2 → T3 → T1  (形成环，检测到死锁)

检测到死锁后：
  1. 选择一个事务作为牺牲者 (Victim)
  2. 回滚该事务
  3. 释放锁，让其他事务继续

选择牺牲者的策略：
  • 事务持有的锁最少
  • 事务执行时间最短
  • 事务修改的数据最少
```

#### 死锁预防
```
1. 按顺序加锁
   ✓ 总是先锁 id 小的记录
   T1: LOCK(1) → LOCK(2)
   T2: LOCK(1) → LOCK(2)
   不会死锁

2. 超时机制
   SET innodb_lock_wait_timeout = 50;
   等待锁超过 50 秒后自动回滚

3. 减少事务粒度
   ❌ START TRANSACTION;
      SELECT ... FOR UPDATE;  -- 锁很多行
      做复杂计算...
      UPDATE ...;
      COMMIT;

   ✓ 先查询，计算后再开事务
      SELECT ...;
      做复杂计算...
      START TRANSACTION;
      UPDATE ...;
      COMMIT;

4. 使用乐观锁
   -- 不加锁，用版本号控制
   SELECT balance, version FROM account WHERE id = 1;
   -- 计算...
   UPDATE account
   SET balance = new_balance, version = version + 1
   WHERE id = 1 AND version = old_version;
```

## 🎯 最佳实践

### 1. 事务设计原则
```sql
-- ✓ 好的事务：短小精悍
START TRANSACTION;
  UPDATE account SET balance = balance - 100 WHERE id = 1;
  UPDATE account SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- ❌ 差的事务：太长
START TRANSACTION;
  SELECT * FROM orders;  -- 大量数据
  -- 复杂业务逻辑处理...
  -- 网络请求...
  -- 文件操作...
  UPDATE inventory SET stock = stock - 1;
COMMIT;

原则：
  1. 事务尽可能短
  2. 不在事务中做耗时操作（网络、文件 IO）
  3. 不在事务中做复杂计算
  4. 按相同顺序访问资源（避免死锁）
```

### 2. 锁使用技巧
```sql
-- 场景 1：库存扣减（需要先检查再更新）
-- ✓ 使用 FOR UPDATE
START TRANSACTION;
SELECT stock FROM inventory WHERE product_id = 1 FOR UPDATE;
IF (stock >= qty) THEN
  UPDATE inventory SET stock = stock - qty WHERE product_id = 1;
  COMMIT;
ELSE
  ROLLBACK;
END IF;

-- 场景 2：读取一致性数据（不需要修改）
-- ✓ 使用 LOCK IN SHARE MODE
SELECT SUM(balance) FROM account LOCK IN SHARE MODE;

-- 场景 3：高并发读取
-- ✓ 使用普通 SELECT（MVCC）
SELECT * FROM products WHERE category = 'phone';
```

### 3. 隔离级别设置
```sql
-- 全局设置
SET GLOBAL TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 会话级设置
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 单个事务设置
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
START TRANSACTION;
...
COMMIT;

-- 查看当前隔离级别
SELECT @@transaction_isolation;  -- MySQL 8.0+
SELECT @@tx_isolation;           -- MySQL 5.x
```

### 4. 处理死锁
```sql
-- 方案 1：捕获异常并重试
BEGIN
  START TRANSACTION;
    UPDATE account SET balance = balance - 100 WHERE id = 1;
    UPDATE account SET balance = balance + 100 WHERE id = 2;
  COMMIT;
EXCEPTION WHEN DEADLOCK_DETECTED THEN
  ROLLBACK;
  -- 等待随机时间后重试
  SLEEP(RANDOM() * 1000);
  RETRY;
END;

-- 方案 2：按顺序加锁
START TRANSACTION;
  -- 总是先锁 id 小的
  IF id1 < id2 THEN
    UPDATE account SET balance = ... WHERE id = id1;
    UPDATE account SET balance = ... WHERE id = id2;
  ELSE
    UPDATE account SET balance = ... WHERE id = id2;
    UPDATE account SET balance = ... WHERE id = id1;
  END IF;
COMMIT;
```

## 🔗 与其他概念的联系

### 与操作系统
- **锁机制** - 类似进程同步的互斥锁、读写锁
- **死锁** - 操作系统的死锁问题
- **MVCC** - 类似写时复制 (Copy-on-Write)

参考：`systems/operating-systems/process-synchronization.md`

### 与数据库其他概念
- [关系模型](./relational-model-sql.md) - 事务操作的对象
- [索引](./indexing-query-optimization.md) - 锁的粒度与索引有关
- [存储引擎](./storage-engines.md) - 日志系统实现 ACID

## 📚 扩展阅读

### 高级主题
- **乐观并发控制 (OCC)** - 无锁并发控制
- **时间戳排序 (Timestamp Ordering)** - 另一种并发控制方法
- **快照隔离 (Snapshot Isolation)** - 介于 RC 和 RR 之间
- **分布式事务** - 2PC、3PC、Saga

### 经典论文
- "A Critique of ANSI SQL Isolation Levels" (1995)
- "Granularity of Locks and Degrees of Consistency" (1976)

---

**掌握事务和并发控制，你就能写出正确、高效的数据库应用！** 🔒
