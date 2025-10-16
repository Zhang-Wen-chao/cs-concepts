# 数据库基础概念 (Database Fundamentals)

> 什么是数据库？为什么需要数据库？

## 🎯 核心概念

**数据库 = 有组织地存储和管理数据的系统**

### 关键问题
- 为什么不直接用文件存储数据？
- 数据库管理系统提供了什么？
- 如何组织和访问数据？

---

## 1️⃣ 为什么需要数据库？

### 文件系统的局限

```
使用普通文件存储数据的问题：

1. 数据冗余和不一致
   users.txt:     Alice, 25, alice@email.com
   orders.txt:    Alice, 25, Order#123
   → Alice 的年龄存了两次，可能不一致

2. 数据访问困难
   • 需要为每个查询写代码
   • 查找 "年龄大于 25 的用户" → 写循环读文件

3. 数据隔离问题
   • 数据分散在多个文件中
   • 难以维护数据关系

4. 完整性问题
   • 年龄必须 >= 0
   • 邮箱必须唯一
   → 需要在每个程序中手动检查

5. 原子性问题
   • 转账：A 账户 -100，B 账户 +100
   • 如果中间崩溃？钱消失了！

6. 并发访问异常
   • 两个程序同时写文件
   • 数据可能损坏

7. 安全性问题
   • 文件系统权限粗粒度
   • 难以实现细粒度访问控制
```

### 数据库的解决方案

```
数据库管理系统 (DBMS) 提供：

✅ 数据抽象
   → 隐藏存储细节，提供统一接口

✅ 数据独立性
   → 修改存储结构不影响应用

✅ 高效数据访问
   → 索引、查询优化

✅ 数据完整性
   → 约束、触发器

✅ 并发控制
   → 事务、锁

✅ 故障恢复
   → 日志、备份

✅ 安全性
   → 用户权限、加密
```

---

## 2️⃣ 数据库系统架构

### 三级模式结构

```
┌─────────────────────────────────────────────┐
│           外模式 (External Schema)          │
│         用户看到的数据视图                   │
│  ┌───────────┐  ┌───────────┐  ┌─────────┐ │
│  │  View 1   │  │  View 2   │  │  View 3 │ │
│  │  用户A    │  │  用户B    │  │  应用C  │ │
│  └───────────┘  └───────────┘  └─────────┘ │
└──────────────────┬──────────────────────────┘
                   │ 外模式/模式映射
┌──────────────────▼──────────────────────────┐
│           概念模式 (Conceptual Schema)       │
│         数据库的逻辑结构                     │
│  ┌──────────────────────────────────────┐  │
│  │  • 表结构定义                        │  │
│  │  • 完整性约束                        │  │
│  │  • 数据关系                          │  │
│  │  • 访问控制                          │  │
│  └──────────────────────────────────────┘  │
└──────────────────┬──────────────────────────┘
                   │ 模式/内模式映射
┌──────────────────▼──────────────────────────┐
│           内模式 (Internal Schema)           │
│         数据的物理存储结构                   │
│  ┌──────────────────────────────────────┐  │
│  │  • 存储文件组织                      │  │
│  │  • 索引结构                          │  │
│  │  • 数据压缩                          │  │
│  │  • 加密                              │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                   │
               ┌───▼───┐
               │  磁盘  │
               └───────┘
```

### 数据独立性

#### 1. 物理独立性
```
修改内模式，不影响概念模式和外模式

例子：
  • 原来：数据存储在机械硬盘
  • 改为：数据存储在 SSD
  • 应用程序无需修改！

  • 原来：使用 B 树索引
  • 改为：使用 B+ 树索引
  • SQL 查询无需修改！
```

#### 2. 逻辑独立性
```
修改概念模式，尽量不影响外模式

例子：
  • 添加新字段到表
  • 旧的查询仍然有效

  • 拆分表
  • 通过视图保持兼容性
```

---

## 3️⃣ 数据模型

### 什么是数据模型？

**数据模型 = 描述数据、数据关系、数据语义和约束的工具**

### 1. 层次模型 (Hierarchical Model)

```
树状结构，每个节点只有一个父节点

例子：文件系统
           公司
            │
    ┌───────┴───────┐
    │               │
  部门A           部门B
    │               │
┌───┴───┐       ┌───┴───┐
员工1  员工2    员工3  员工4

优点：结构清晰
缺点：不适合多对多关系
```

### 2. 网状模型 (Network Model)

```
图结构，允许多个父节点

例子：课程-学生关系
    课程A ────┐
      │      │
      │   ┌──▼──┐
      └──→│ 学生1│
          └──┬──┘
    课程B ───┘

优点：灵活
缺点：复杂，难以维护
```

### 3. 关系模型 (Relational Model)

**最成功的数据模型！**

```
数据组织成表（关系）

Students 表：
┌────┬──────┬─────┬─────────────────┐
│ ID │ Name │ Age │     Email       │
├────┼──────┼─────┼─────────────────┤
│  1 │Alice │  20 │alice@email.com  │
│  2 │  Bob │  22 │bob@email.com    │
│  3 │Carol │  21 │carol@email.com  │
└────┴──────┴─────┴─────────────────┘

Courses 表：
┌────┬─────────────┬─────────┐
│ ID │    Name     │ Credits │
├────┼─────────────┼─────────┤
│ 10 │  Database   │    3    │
│ 11 │  Algorithms │    4    │
└────┴─────────────┴─────────┘

Enrollments 表（多对多关系）：
┌────────────┬───────────┬───────┐
│ Student_ID │ Course_ID │ Grade │
├────────────┼───────────┼───────┤
│      1     │     10    │   A   │
│      1     │     11    │   B   │
│      2     │     10    │   A   │
└────────────┴───────────┴───────┘

特点：
  • 简单直观（表格）
  • 数学基础（关系代数）
  • 数据独立性好
  • 支持复杂查询
```

### 4. 对象模型 (Object Model)

```
面向对象的数据库

class Student {
    int id;
    String name;
    int age;
    List<Course> courses;  // 直接存储对象引用
}

优点：与 OOP 语言天然集成
缺点：查询性能不如关系型
```

### 5. NoSQL 模型

```
键值存储 (Redis):
  key: "user:1001"
  value: {"name": "Alice", "age": 20}

文档存储 (MongoDB):
  {
    "_id": "1001",
    "name": "Alice",
    "age": 20,
    "courses": [
      {"name": "Database", "grade": "A"},
      {"name": "Algorithms", "grade": "B"}
    ]
  }

列族存储 (Cassandra):
  Row Key: "user:1001"
  Columns: name="Alice", age=20, ...

图数据库 (Neo4j):
  (Alice) -[:ENROLLED_IN {grade: "A"}]-> (Database)
  (Alice) -[:FRIENDS_WITH]-> (Bob)
```

---

## 4️⃣ 数据库管理系统 (DBMS)

### DBMS 的组件

```
┌─────────────────────────────────────────────┐
│           用户/应用程序                      │
└──────────────────┬──────────────────────────┘
                   │ SQL / API
┌──────────────────▼──────────────────────────┐
│        查询处理器 (Query Processor)          │
│  ┌──────────────────────────────────────┐  │
│  │  • SQL 解析器 (Parser)               │  │
│  │  • 查询优化器 (Optimizer)            │  │
│  │  • 执行引擎 (Execution Engine)       │  │
│  └──────────────────────────────────────┘  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│        存储管理器 (Storage Manager)          │
│  ┌──────────────────────────────────────┐  │
│  │  • 事务管理器 (Transaction Manager)  │  │
│  │  • 缓冲管理器 (Buffer Manager)       │  │
│  │  • 文件管理器 (File Manager)         │  │
│  │  • 索引管理器 (Index Manager)        │  │
│  └──────────────────────────────────────┘  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              磁盘存储                        │
│  数据文件 | 索引文件 | 日志文件             │
└─────────────────────────────────────────────┘
```

### 查询处理流程

```
SQL 查询: SELECT name FROM students WHERE age > 20;

1. 解析 (Parsing)
   ↓
   语法树 (Parse Tree)

2. 查询重写 (Rewrite)
   ↓
   优化的语法树

3. 查询优化 (Optimization)
   ↓
   执行计划 (Query Plan)
   • 选择索引
   • 选择连接算法
   • 估算代价

4. 执行 (Execution)
   ↓
   结果集

示例执行计划：
  1. Index Scan on students (age > 20)
  2. Projection (name)
  3. Return results
```

---

## 5️⃣ ACID 特性

### 事务的四大特性

```
事务 (Transaction) = 数据库操作的基本单位

ACID:
  • Atomicity (原子性)
  • Consistency (一致性)
  • Isolation (隔离性)
  • Durability (持久性)
```

### 1. 原子性 (Atomicity)

```
事务要么全部完成，要么全部不做

例子：银行转账
  BEGIN TRANSACTION;
    UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
    UPDATE accounts SET balance = balance + 100 WHERE id = 'B';
  COMMIT;

如果中间崩溃：
  ❌ A 扣了钱，B 没收到 → 违反原子性
  ✅ 回滚到初始状态 → 保证原子性

实现：Undo Log（撤销日志）
```

### 2. 一致性 (Consistency)

```
事务执行前后，数据库保持一致状态

例子：转账前后，总金额不变
  A 账户: 1000
  B 账户: 500
  总金额: 1500

  转账 100 后：
  A 账户: 900
  B 账户: 600
  总金额: 1500  ✅ 一致

约束：
  • 主键唯一
  • 外键有效
  • 业务规则（如余额 >= 0）
```

### 3. 隔离性 (Isolation)

```
并发事务之间互不干扰

例子：两个人同时转账
  事务1: A → B 转 100
  事务2: A → C 转 50

  如果没有隔离：
    两个事务都读到 A = 1000
    事务1: A = 1000 - 100 = 900 写回
    事务2: A = 1000 - 50 = 950 写回
    → 最终 A = 950（错误！应该是 850）

  有隔离：
    事务按顺序执行或使用锁
    → 结果正确
```

### 4. 持久性 (Durability)

```
事务提交后，数据永久保存

例子：
  COMMIT 成功后，即使系统崩溃，数据也不丢失

实现：
  • Redo Log（重做日志）
  • WAL (Write-Ahead Logging)
  • 先写日志，再写数据
```

---

## 6️⃣ 数据完整性

### 1. 实体完整性

```
主键约束：主键不能为 NULL，且唯一

CREATE TABLE students (
    id INT PRIMARY KEY,        -- 主键
    name VARCHAR(100) NOT NULL
);

INSERT INTO students VALUES (1, 'Alice');
INSERT INTO students VALUES (1, 'Bob');    -- ❌ 错误：主键重复
INSERT INTO students VALUES (NULL, 'Eve'); -- ❌ 错误：主键不能为 NULL
```

### 2. 参照完整性

```
外键约束：外键必须引用已存在的主键

CREATE TABLE enrollments (
    student_id INT,
    course_id INT,
    FOREIGN KEY (student_id) REFERENCES students(id),
    FOREIGN KEY (course_id) REFERENCES courses(id)
);

INSERT INTO enrollments VALUES (999, 10);  -- ❌ 学生 999 不存在
```

### 3. 用户定义完整性

```
业务规则约束

CREATE TABLE employees (
    id INT PRIMARY KEY,
    age INT CHECK (age >= 18 AND age <= 65),  -- 年龄约束
    salary DECIMAL(10,2) CHECK (salary > 0),  -- 工资必须为正
    email VARCHAR(100) UNIQUE                  -- 邮箱唯一
);
```

---

## 7️⃣ 数据库语言

### 1. 数据定义语言 (DDL)

```sql
-- 创建表
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

-- 修改表结构
ALTER TABLE users ADD COLUMN email VARCHAR(100);

-- 删除表
DROP TABLE users;

-- 创建索引
CREATE INDEX idx_name ON users(name);
```

### 2. 数据操作语言 (DML)

```sql
-- 查询
SELECT * FROM users WHERE age > 20;

-- 插入
INSERT INTO users VALUES (1, 'Alice', 20);

-- 更新
UPDATE users SET age = 21 WHERE id = 1;

-- 删除
DELETE FROM users WHERE id = 1;
```

### 3. 数据控制语言 (DCL)

```sql
-- 授权
GRANT SELECT, INSERT ON users TO user1;

-- 撤销权限
REVOKE INSERT ON users FROM user1;
```

### 4. 事务控制语言 (TCL)

```sql
-- 开始事务
BEGIN TRANSACTION;

-- 提交
COMMIT;

-- 回滚
ROLLBACK;

-- 保存点
SAVEPOINT sp1;
ROLLBACK TO sp1;
```

---

## 8️⃣ 实际应用

### 数据库 vs 文件系统对比

```
场景：存储用户信息

文件系统方式：
  users.txt:
    1,Alice,20,alice@email.com
    2,Bob,22,bob@email.com

  查询年龄 > 20 的用户：
    ❌ 需要读取整个文件
    ❌ 需要编写解析代码
    ❌ 并发访问需要手动加锁

数据库方式：
  CREATE TABLE users (...);
  SELECT * FROM users WHERE age > 20;

  ✅ 自动使用索引
  ✅ SQL 简洁
  ✅ 自动处理并发
```

### 何时使用文件 vs 数据库

```
使用文件：
  • 配置文件（config.json）
  • 日志文件（app.log）
  • 临时数据
  • 数据量小且简单

使用数据库：
  • 结构化数据
  • 需要复杂查询
  • 需要事务保证
  • 并发访问多
  • 数据量大
```

---

## 🔗 与其他概念的联系

### 与操作系统
- **文件系统** - DBMS 底层使用文件存储
- **进程管理** - 数据库服务器是多进程/多线程程序
- **内存管理** - 缓冲池管理类似虚拟内存

参考：`systems/operating-systems/`

### 与数据结构
- **B+ 树** - 索引实现
- **哈希表** - 内存结构、哈希索引
- **队列** - 事务处理、查询队列

参考：`fundamentals/data-structures/`

### 与网络
- **客户端-服务器** - 数据库连接模型
- **协议** - MySQL 协议、PostgreSQL 协议

参考：`systems/networks/`

---

## 📚 深入学习

### 推荐资源
- *Database System Concepts* - Silberschatz (教材)
- CMU 15-445: Database Systems (课程)
- MySQL/PostgreSQL 官方文档

### 实践项目
- 设计一个小型数据库应用（如图书管理系统）
- 分析真实应用的数据库设计
- 尝试不同的数据模型（关系型 vs 文档型）

### 下一步
- [关系模型与 SQL](./relational-model-sql.md) - 深入关系型数据库
- [索引与查询优化](./indexing-query-optimization.md) - 提升查询性能
- [事务与并发控制](./transactions-concurrency.md) - 保证数据一致性

---

**理解数据库基础，就掌握了数据管理的核心思想！** 📊
