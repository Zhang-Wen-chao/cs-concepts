# 关系模型与 SQL (Relational Model and SQL)

> 关系型数据库的理论基础和实践语言

## 🎯 核心思想

关系模型是数据库领域最成功的理论之一，由 Edgar F. Codd 在 1970 年提出。它用简单的**表格**来组织数据，配合强大的 **SQL 语言**进行操作。

**一句话理解：**
- 关系模型：用**二维表**表示数据和数据之间的关系
- SQL：操作这些表的**标准语言**

## 📖 关系模型基础

### 1. 核心概念

#### 关系 (Relation) = 表 (Table)
```
关系就是一张二维表：

员工表 (employees)
┌────┬──────┬─────────┬────────┐
│ id │ name │ dept_id │ salary │
├────┼──────┼─────────┼────────┤
│ 1  │ 张三 │ 101     │ 8000   │
│ 2  │ 李四 │ 102     │ 9000   │
│ 3  │ 王五 │ 101     │ 7500   │
└────┴──────┴─────────┴────────┘

• 每一行 = 元组 (Tuple) = 记录
• 每一列 = 属性 (Attribute) = 字段
• 表名 = 关系名
```

#### 域 (Domain)
```
每个属性的取值范围：

name: 字符串类型，最长 50 字符
salary: 整数，范围 0-999999
dept_id: 整数，必须存在于部门表中
```

#### 键 (Key)

**1. 超键 (Super Key)**
```
能唯一标识一条记录的属性集合

员工表的超键：
  ✓ {id}
  ✓ {id, name}
  ✓ {id, name, salary}
  ✓ {身份证号}

超键可能包含冗余属性
```

**2. 候选键 (Candidate Key)**
```
最小的超键（去掉任何属性都不再是超键）

员工表的候选键：
  ✓ {id}
  ✓ {身份证号}

都是候选键，但通常选 id 作为主键
```

**3. 主键 (Primary Key)**
```sql
-- 从候选键中选一个作为主键
CREATE TABLE employees (
    id INT PRIMARY KEY,        -- 主键
    id_card VARCHAR(18) UNIQUE, -- 候选键
    name VARCHAR(50),
    salary INT
);

主键特点：
  • 唯一性：不能重复
  • 非空性：不能为 NULL
  • 稳定性：一般不会改变
```

**4. 外键 (Foreign Key)**
```sql
-- 外键建立表之间的关系
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES departments(id)
);

CREATE TABLE departments (
    id INT PRIMARY KEY,
    name VARCHAR(50)
);

外键作用：
  • 维护引用完整性
  • 不能插入不存在的 dept_id
  • 删除部门时检查是否有员工
```

### 2. 完整性约束

#### 实体完整性
```sql
-- 主键不能为空
INSERT INTO employees (id, name) VALUES (NULL, '张三');  -- ❌ 错误

INSERT INTO employees (id, name) VALUES (1, '张三');     -- ✓ 正确
```

#### 参照完整性
```sql
-- 外键必须引用存在的值
INSERT INTO employees (id, name, dept_id)
VALUES (1, '张三', 999);  -- ❌ 错误：部门 999 不存在

INSERT INTO employees (id, name, dept_id)
VALUES (1, '张三', 101);  -- ✓ 正确：部门 101 存在
```

#### 用户定义完整性
```sql
-- 自定义约束
CREATE TABLE employees (
    id INT PRIMARY KEY,
    age INT CHECK (age >= 18 AND age <= 65),  -- 年龄范围
    salary INT CHECK (salary > 0),             -- 工资必须为正
    email VARCHAR(100) UNIQUE                  -- 邮箱唯一
);
```

## 🔧 关系代数

关系代数是关系模型的理论基础，SQL 是它的实现。

### 1. 选择 (Selection) - σ
```
从表中选择满足条件的行

σ(salary > 8000)(employees)

┌────┬──────┬────────┐
│ id │ name │ salary │
├────┼──────┼────────┤
│ 2  │ 李四 │ 9000   │
│ 5  │ 赵六 │ 8500   │
└────┴──────┴────────┘

SQL 等价：
SELECT * FROM employees WHERE salary > 8000;
```

### 2. 投影 (Projection) - π
```
从表中选择某些列

π(id, name)(employees)

┌────┬──────┐
│ id │ name │
├────┼──────┤
│ 1  │ 张三 │
│ 2  │ 李四 │
│ 3  │ 王五 │
└────┴──────┘

SQL 等价：
SELECT id, name FROM employees;
```

### 3. 连接 (Join) - ⋈
```
组合两个表的数据

employees ⋈ departments (通过 dept_id)

┌────┬──────┬────────┬───────────┐
│ id │ name │ salary │ dept_name │
├────┼──────┼────────┼───────────┤
│ 1  │ 张三 │ 8000   │ 技术部    │
│ 2  │ 李四 │ 9000   │ 销售部    │
└────┴──────┴────────┴───────────┘

SQL 等价：
SELECT e.*, d.name as dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.id;
```

### 4. 并、交、差
```
集合运算

并 (Union): 两个查询结果的合集
交 (Intersection): 两个查询结果的交集
差 (Difference): 在 A 中但不在 B 中

SQL 示例：
SELECT name FROM employees_2023
UNION
SELECT name FROM employees_2024;
```

## 💻 SQL 语言

### 1. 数据定义语言 (DDL)

#### 创建表
```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    age INT CHECK (age >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
);
```

#### 修改表
```sql
-- 添加列
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 删除列
ALTER TABLE users DROP COLUMN age;

-- 修改列
ALTER TABLE users MODIFY COLUMN username VARCHAR(100);

-- 添加索引
ALTER TABLE users ADD INDEX idx_phone (phone);

-- 添加外键
ALTER TABLE orders
ADD CONSTRAINT fk_user
FOREIGN KEY (user_id) REFERENCES users(id);
```

#### 删除表
```sql
DROP TABLE users;  -- 删除表和数据

TRUNCATE TABLE users;  -- 清空数据，保留表结构
```

### 2. 数据查询语言 (DQL)

#### 基本查询
```sql
-- 查询所有列
SELECT * FROM employees;

-- 查询指定列
SELECT id, name, salary FROM employees;

-- 去重
SELECT DISTINCT dept_id FROM employees;

-- 条件查询
SELECT * FROM employees WHERE salary > 8000;

-- 多条件
SELECT * FROM employees
WHERE salary > 8000 AND dept_id = 101;

-- 模糊查询
SELECT * FROM employees WHERE name LIKE '张%';

-- 范围查询
SELECT * FROM employees WHERE salary BETWEEN 7000 AND 9000;

-- IN 查询
SELECT * FROM employees WHERE dept_id IN (101, 102, 103);
```

#### 排序和分页
```sql
-- 排序
SELECT * FROM employees ORDER BY salary DESC;

-- 多字段排序
SELECT * FROM employees
ORDER BY dept_id ASC, salary DESC;

-- 分页
SELECT * FROM employees
ORDER BY id
LIMIT 10 OFFSET 20;  -- 跳过 20 条，取 10 条

-- 简化写法
SELECT * FROM employees
ORDER BY id
LIMIT 20, 10;  -- 从第 20 条开始，取 10 条
```

#### 聚合函数
```sql
-- 统计
SELECT COUNT(*) FROM employees;
SELECT COUNT(DISTINCT dept_id) FROM employees;

-- 求和、平均、最大、最小
SELECT
    SUM(salary) as total_salary,
    AVG(salary) as avg_salary,
    MAX(salary) as max_salary,
    MIN(salary) as min_salary
FROM employees;

-- 分组聚合
SELECT dept_id, COUNT(*) as emp_count, AVG(salary) as avg_salary
FROM employees
GROUP BY dept_id;

-- 分组过滤
SELECT dept_id, AVG(salary) as avg_salary
FROM employees
GROUP BY dept_id
HAVING AVG(salary) > 8000;
```

#### 连接查询

**内连接 (INNER JOIN)**
```sql
-- 只返回两表都有匹配的记录
SELECT e.name, d.name as dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;

employees:            departments:         结果:
┌────┬──────┬─────┐   ┌────┬──────┐      ┌──────┬──────┐
│ id │ name │dept │   │ id │ name │      │ name │ dept │
├────┼──────┼─────┤   ├────┼──────┤      ├──────┼──────┤
│ 1  │ 张三 │ 101 │   │101 │ 技术 │  →   │ 张三 │ 技术 │
│ 2  │ 李四 │ 102 │   │102 │ 销售 │      │ 李四 │ 销售 │
│ 3  │ 王五 │ 103 │   │103 │ 人事 │      │ 王五 │ 人事 │
│ 4  │ 赵六 │NULL │   └────┴──────┘      └──────┴──────┘
└────┴──────┴─────┘   赵六被过滤（没有部门）
```

**左外连接 (LEFT JOIN)**
```sql
-- 返回左表所有记录，右表没有匹配则为 NULL
SELECT e.name, d.name as dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;

结果:
┌──────┬──────┐
│ name │ dept │
├──────┼──────┤
│ 张三 │ 技术 │
│ 李四 │ 销售 │
│ 王五 │ 人事 │
│ 赵六 │ NULL │  ← 保留（没有部门也显示）
└──────┴──────┘
```

**右外连接 (RIGHT JOIN)**
```sql
-- 返回右表所有记录，左表没有匹配则为 NULL
SELECT e.name, d.name as dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;

如果 departments 有 104（财务部），但没有员工：
┌──────┬──────┐
│ name │ dept │
├──────┼──────┤
│ 张三 │ 技术 │
│ 李四 │ 销售 │
│ 王五 │ 人事 │
│ NULL │ 财务 │  ← 保留（没有员工也显示）
└──────┴──────┘
```

**全外连接 (FULL OUTER JOIN)**
```sql
-- 返回两表所有记录，没有匹配则为 NULL
-- MySQL 不直接支持，需要用 UNION 模拟
SELECT e.name, d.name as dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
UNION
SELECT e.name, d.name as dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;
```

**自连接**
```sql
-- 查找同一部门的其他员工
SELECT e1.name, e2.name as colleague
FROM employees e1
JOIN employees e2 ON e1.dept_id = e2.dept_id
WHERE e1.id != e2.id;
```

#### 子查询

**WHERE 子查询**
```sql
-- 查找工资高于平均工资的员工
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- IN 子查询
SELECT name FROM employees
WHERE dept_id IN (
    SELECT id FROM departments WHERE name LIKE '技术%'
);

-- EXISTS 子查询
SELECT name FROM employees e
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = e.id
);
```

**FROM 子查询**
```sql
-- 子查询作为临时表
SELECT dept_id, avg_salary
FROM (
    SELECT dept_id, AVG(salary) as avg_salary
    FROM employees
    GROUP BY dept_id
) as dept_avg
WHERE avg_salary > 8000;
```

**SELECT 子查询**
```sql
-- 查询每个员工及其部门平均工资
SELECT
    name,
    salary,
    (SELECT AVG(salary)
     FROM employees e2
     WHERE e2.dept_id = e1.dept_id) as dept_avg
FROM employees e1;
```

### 3. 数据操作语言 (DML)

#### 插入
```sql
-- 插入单条
INSERT INTO users (username, email, age)
VALUES ('zhangsan', 'zhang@example.com', 25);

-- 插入多条
INSERT INTO users (username, email, age) VALUES
    ('lisi', 'li@example.com', 30),
    ('wangwu', 'wang@example.com', 28);

-- 从查询结果插入
INSERT INTO users_backup
SELECT * FROM users WHERE age > 30;
```

#### 更新
```sql
-- 更新单条
UPDATE users SET age = 26 WHERE username = 'zhangsan';

-- 更新多列
UPDATE users
SET age = 26, email = 'new@example.com'
WHERE username = 'zhangsan';

-- 批量更新
UPDATE users SET age = age + 1 WHERE age < 30;

-- 使用子查询更新
UPDATE employees SET salary = salary * 1.1
WHERE dept_id IN (
    SELECT id FROM departments WHERE name = '技术部'
);
```

#### 删除
```sql
-- 删除单条
DELETE FROM users WHERE id = 1;

-- 批量删除
DELETE FROM users WHERE age < 18;

-- 使用子查询删除
DELETE FROM orders WHERE user_id IN (
    SELECT id FROM users WHERE status = 'inactive'
);
```

### 4. SQL 执行顺序

```sql
SELECT DISTINCT column, AGG_FUNC(column)
FROM table1
JOIN table2 ON condition
WHERE condition
GROUP BY column
HAVING condition
ORDER BY column
LIMIT count OFFSET start;

实际执行顺序：
1. FROM     - 确定数据来源
2. JOIN     - 连接表
3. WHERE    - 过滤行
4. GROUP BY - 分组
5. HAVING   - 过滤分组
6. SELECT   - 选择列
7. DISTINCT - 去重
8. ORDER BY - 排序
9. LIMIT    - 限制结果数量
```

**示例分析：**
```sql
SELECT dept_id, AVG(salary) as avg_salary
FROM employees
WHERE salary > 5000
GROUP BY dept_id
HAVING AVG(salary) > 8000
ORDER BY avg_salary DESC
LIMIT 3;

执行过程：
1. FROM employees           → 获取员工表
2. WHERE salary > 5000      → 过滤低工资员工
3. GROUP BY dept_id         → 按部门分组
4. HAVING AVG(salary)>8000  → 过滤平均工资低的部门
5. SELECT dept_id, AVG...   → 计算并选择列
6. ORDER BY avg_salary DESC → 按平均工资降序
7. LIMIT 3                  → 取前 3 个
```

## 📐 范式理论

范式是设计良好数据库的指导原则，用于减少数据冗余和避免异常。

### 范式之间的关系

范式是**层层递进、逐步强化**的关系：

```
1NF ⊂ 2NF ⊂ 3NF ⊂ BCNF ⊂ 4NF ⊂ 5NF
```

- 满足高级范式必然满足低级范式
- 范式级别越高，冗余越少，但查询可能需要更多 JOIN
- 实际应用中，大部分场景满足 **3NF 或 BCNF** 就足够

### 第一范式 (1NF) - 原子性
```
每个字段都是不可分割的原子值

❌ 违反 1NF：
┌────┬──────┬──────────────┐
│ id │ name │ phones       │
├────┼──────┼──────────────┤
│ 1  │ 张三 │ 123,456,789  │  ← 包含多个电话
└────┴──────┴──────────────┘

✓ 符合 1NF：
┌────┬──────┬───────┐
│ id │ name │ phone │
├────┼──────┼───────┤
│ 1  │ 张三 │ 123   │
│ 1  │ 张三 │ 456   │
│ 1  │ 张三 │ 789   │
└────┴──────┴───────┘
```

### 第二范式 (2NF) - 完全依赖
```
非主键字段必须完全依赖于主键（不能只依赖主键的一部分）

❌ 违反 2NF：
订单明细表（联合主键：order_id, product_id）
┌──────────┬────────────┬──────┬──────────────┐
│ order_id │ product_id │ qty  │ product_name │
├──────────┼────────────┼──────┼──────────────┤
│ 1001     │ 101        │ 2    │ iPhone       │
└──────────┴────────────┴──────┴──────────────┘
         ↑                        ↑
         主键                     只依赖 product_id
                                  (部分依赖)

✓ 符合 2NF：拆分成两个表
订单明细表：
┌──────────┬────────────┬──────┐
│ order_id │ product_id │ qty  │
└──────────┴────────────┴──────┘

产品表：
┌────────────┬──────────────┐
│ product_id │ product_name │
└────────────┴──────────────┘
```

### 第三范式 (3NF) - 无传递依赖
```
非主键字段不能依赖于其他非主键字段

❌ 违反 3NF：
员工表
┌────┬──────┬─────────┬───────────┐
│ id │ name │ dept_id │ dept_name │
├────┼──────┼─────────┼───────────┤
│ 1  │ 张三 │ 101     │ 技术部    │
└────┴──────┴─────────┴───────────┘
  ↑            ↑          ↑
  主键         非主键     非主键
                ↓          ↓
              dept_name 依赖于 dept_id
              (传递依赖)

✓ 符合 3NF：拆分成两个表
员工表：
┌────┬──────┬─────────┐
│ id │ name │ dept_id │
└────┴──────┴─────────┘

部门表：
┌─────────┬───────────┐
│ dept_id │ dept_name │
└─────────┴───────────┘
```

### BCNF (Boyce-Codd 范式)
```
更严格的 3NF：所有函数依赖的决定因素都必须是候选键

示例：教师-课程-教室
┌─────────┬────────┬───────────┐
│ teacher │ course │ classroom │
├─────────┼────────┼───────────┤
│ 张老师  │ 数学   │ A101      │
│ 张老师  │ 物理   │ A101      │
│ 李老师  │ 数学   │ B202      │
└─────────┴────────┴───────────┘

问题：
• 候选键：(teacher, course)
• 但 teacher → classroom (每个老师固定教室)
• classroom 依赖于非候选键的一部分

解决：拆分表
┌─────────┬────────┐   ┌─────────┬───────────┐
│ teacher │ course │   │ teacher │ classroom │
└─────────┴────────┘   └─────────┴───────────┘
```

### 第四范式 (4NF) - 消除多值依赖
```
消除非平凡且非函数依赖的多值依赖

示例：教师-课程-书籍
┌─────────┬────────┬──────────┐
│ teacher │ course │ book     │
├─────────┼────────┼──────────┤
│ 张老师  │ 数学   │ 教材A    │
│ 张老师  │ 数学   │ 教材B    │
│ 张老师  │ 物理   │ 教材A    │
│ 张老师  │ 物理   │ 教材B    │
└─────────┴────────┴──────────┘

问题：
• 张老师教{数学, 物理}，推荐{教材A, 教材B}
• 课程和书籍相互独立，但产生笛卡尔积
• 导致冗余：课程和书籍的所有组合都要存储

解决：拆分表
┌─────────┬────────┐   ┌─────────┬──────────┐
│ teacher │ course │   │ teacher │ book     │
├─────────┼────────┤   ├─────────┼──────────┤
│ 张老师  │ 数学   │   │ 张老师  │ 教材A    │
│ 张老师  │ 物理   │   │ 张老师  │ 教材B    │
└─────────┴────────┘   └─────────┴──────────┘
```

### 第五范式 (5NF/PJNF) - 消除连接依赖
```
表不能被无损分解为更小的表（较少使用）

实际应用中，4NF 和 5NF 很少需要，多数场景 3NF/BCNF 已足够。
```

### 范式选择与反范式化

#### 范式选择原则
```
1NF：最低要求，几乎所有表都应满足
2NF/3NF：标准设计，大多数业务表应满足
BCNF：严格设计，重要的核心表应满足
4NF/5NF：特殊场景，一般不需要
```

#### 反范式化
```
有时为了性能，故意违反范式

示例：电商订单
规范化设计：
  orders 表 + order_items 表 + products 表
  查询订单需要 3 次 JOIN

反范式化设计：
  在 order_items 中冗余存储 product_name、price
  查询订单只需 1 次 JOIN

权衡：
  ✓ 查询更快
  ✓ 代码更简单
  ❌ 更新时需要维护多处数据
  ❌ 占用更多存储空间

何时反范式化：
  • 读多写少的场景
  • 冗余数据很少变化
  • 查询性能是瓶颈
```

## 🎯 最佳实践

### 1. 表设计原则
```sql
-- ✓ 好的设计
CREATE TABLE users (
    -- 使用自增主键
    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    -- 必填字段标记 NOT NULL
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,

    -- 添加唯一索引
    UNIQUE KEY uk_username (username),
    UNIQUE KEY uk_email (email),

    -- 添加业务索引
    INDEX idx_created_at (created_at),

    -- 使用时间戳记录创建和更新时间
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                         ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ❌ 差的设计
CREATE TABLE users (
    -- 使用 UUID 作为主键（性能差）
    id VARCHAR(36) PRIMARY KEY,

    -- 没有标记 NOT NULL
    username VARCHAR(50),

    -- 没有索引（查询慢）

    -- 使用 DATETIME 而非 TIMESTAMP
    created_at DATETIME
);
```

### 2. 查询优化技巧
```sql
-- ✓ 使用索引
SELECT * FROM users WHERE username = 'zhangsan';  -- username 有索引

-- ❌ 破坏索引
SELECT * FROM users WHERE LOWER(username) = 'zhangsan';  -- 函数破坏索引

-- ✓ 只查询需要的列
SELECT id, username FROM users WHERE age > 18;

-- ❌ 查询所有列
SELECT * FROM users WHERE age > 18;

-- ✓ 使用 LIMIT
SELECT * FROM users ORDER BY created_at DESC LIMIT 100;

-- ❌ 不限制结果数量
SELECT * FROM users ORDER BY created_at DESC;

-- ✓ 使用 JOIN
SELECT u.name, o.total
FROM users u
JOIN orders o ON u.id = o.user_id;

-- ❌ 在应用层拼接
SELECT * FROM users;  -- 取所有用户
SELECT * FROM orders; -- 取所有订单
-- 然后在代码里关联
```

### 3. 事务使用
```sql
-- ✓ 短事务
START TRANSACTION;
UPDATE account SET balance = balance - 100 WHERE id = 1;
UPDATE account SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- ❌ 长事务（持有锁时间长）
START TRANSACTION;
UPDATE account SET balance = balance - 100 WHERE id = 1;
-- 做复杂计算或网络请求...
UPDATE account SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

## 🔗 与其他概念的联系

### 与数据结构
- **B+ 树** - 索引实现的基础
- **哈希表** - 哈希索引的基础
- **链表** - 冲突解决

参考：`fundamentals/data-structures/`

### 与算法
- **查询优化** - 动态规划选择最优执行计划
- **连接算法** - 嵌套循环、哈希连接、排序合并

参考：`fundamentals/algorithms/`

### 与其他数据库概念
- [索引](./indexing-query-optimization.md) - 加速查询
- [事务](./transactions-concurrency.md) - 保证一致性
- [存储引擎](./storage-engines.md) - 底层实现

## 📚 扩展阅读

### 关系模型
- Codd 的论文："A Relational Model of Data for Large Shared Data Banks"
- 关系代数与关系演算
- 函数依赖理论

### SQL 标准
- SQL-92、SQL:1999、SQL:2003
- 窗口函数 (Window Functions)
- 公共表表达式 (CTE)
- 递归查询

### 高级主题
- 物化视图
- 分区表
- 全文搜索
- JSON 支持

---

**掌握关系模型和 SQL，你就掌握了数据库的核心！** 🗃️
