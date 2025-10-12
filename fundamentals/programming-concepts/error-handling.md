# Error Handling - 错误处理

> 如何优雅地处理程序中的错误？如何让程序更健壮？

## 🎯 什么是错误？

### 错误的分类

#### 1. 语法错误 (Syntax Error)
代码写错了，无法运行

```python
print("hello"  # ❌ 语法错误：缺少右括号
if x = 5:      # ❌ 语法错误：应该用 ==
```

#### 2. 运行时错误 (Runtime Error)
代码语法正确，但运行时出问题

```python
x = 10 / 0           # ❌ ZeroDivisionError
list = [1, 2, 3]
print(list[10])      # ❌ IndexError
```

#### 3. 逻辑错误 (Logic Error)
代码能运行，但结果不对

```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers) + 1  # ❌ 多加了1
```

**错误处理主要针对：运行时错误**

---

## 🔧 错误处理方式

### 1. 异常机制 (Exception)

#### Python: try-except

```python
try:
    # 可能出错的代码
    result = 10 / 0
except ZeroDivisionError:
    # 处理特定错误
    print("除数不能为零")
except Exception as e:
    # 处理所有其他错误
    print(f"发生错误: {e}")
else:
    # 没有错误时执行
    print("计算成功")
finally:
    # 无论如何都执行（清理资源）
    print("执行完毕")
```

#### 多个异常处理

```python
try:
    file = open("data.txt")
    data = file.read()
    number = int(data)
except FileNotFoundError:
    print("文件不存在")
except ValueError:
    print("数据格式错误")
except Exception as e:
    print(f"未知错误: {e}")
finally:
    if 'file' in locals():
        file.close()
```

#### 主动抛出异常

```python
def withdraw(balance, amount):
    if amount <= 0:
        raise ValueError("金额必须大于0")
    if amount > balance:
        raise ValueError("余额不足")
    return balance - amount

try:
    new_balance = withdraw(100, 150)
except ValueError as e:
    print(f"操作失败: {e}")
```

#### 自定义异常

```python
class InsufficientFundsError(Exception):
    """余额不足异常"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"余额 {balance} 不足以支付 {amount}")

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    withdraw(100, 150)
except InsufficientFundsError as e:
    print(f"错误: {e}")
    print(f"当前余额: {e.balance}, 需要: {e.amount}")
```

---

### 2. 错误码 (Error Code)

#### C语言风格

```c
// 返回错误码
int divide(int a, int b, int* result) {
    if (b == 0) {
        return -1;  // 错误码：除数为零
    }
    *result = a / b;
    return 0;  // 成功
}

// 使用
int result;
int status = divide(10, 2, &result);
if (status == 0) {
    printf("结果: %d\n", result);
} else {
    printf("错误: 除数不能为零\n");
}
```

#### Python中的错误码

```python
def divide(a, b):
    if b == 0:
        return None, "除数不能为零"  # (结果, 错误信息)
    return a / b, None

# 使用
result, error = divide(10, 0)
if error:
    print(f"错误: {error}")
else:
    print(f"结果: {result}")
```

---

### 3. Result类型 (函数式风格)

#### Rust的Result

```rust
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        Err("除数不能为零".to_string())  // 错误
    } else {
        Ok(a / b)  // 成功
    }
}

// 使用
match divide(10, 0) {
    Ok(result) => println!("结果: {}", result),
    Err(error) => println!("错误: {}", error),
}

// 或者用 ? 操作符传播错误
fn calculate() -> Result<i32, String> {
    let result = divide(10, 2)?;  // 如果出错，自动返回错误
    Ok(result * 2)
}
```

#### Python模拟Result

```python
from typing import Union

class Ok:
    def __init__(self, value):
        self.value = value

class Err:
    def __init__(self, error):
        self.error = error

def divide(a, b) -> Union[Ok, Err]:
    if b == 0:
        return Err("除数不能为零")
    return Ok(a / b)

# 使用
result = divide(10, 0)
if isinstance(result, Ok):
    print(f"结果: {result.value}")
else:
    print(f"错误: {result.error}")
```

---

### 4. Option/Maybe类型 (处理空值)

```python
# Python的Optional
from typing import Optional

def find_user(user_id: int) -> Optional[dict]:
    users = {1: {"name": "Alice"}, 2: {"name": "Bob"}}
    return users.get(user_id)  # 找不到返回None

# 使用
user = find_user(3)
if user is not None:
    print(user["name"])
else:
    print("用户不存在")
```

```rust
// Rust的Option
fn find_user(id: i32) -> Option<String> {
    if id == 1 {
        Some("Alice".to_string())  // 找到了
    } else {
        None  // 没找到
    }
}

// 使用
match find_user(1) {
    Some(name) => println!("找到用户: {}", name),
    None => println!("用户不存在"),
}
```

---

## ⚠️ 常见错误类型

### 1. 空值错误

```python
# ❌ 没有检查None
def get_name(user):
    return user["name"]  # 如果user是None，崩溃！

# ✅ 检查None
def get_name(user):
    if user is None:
        return "未知用户"
    return user.get("name", "匿名")
```

### 2. 边界错误

```python
# ❌ 没有检查索引
def get_first(items):
    return items[0]  # 如果列表为空，崩溃！

# ✅ 检查边界
def get_first(items):
    if not items:
        return None
    return items[0]
```

### 3. 类型错误

```python
# ❌ 没有验证类型
def calculate(x, y):
    return x + y  # 如果传入字符串，行为可能不符预期

# ✅ 验证类型
def calculate(x, y):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise TypeError("参数必须是数字")
    return x + y
```

### 4. 资源泄漏

```python
# ❌ 可能忘记关闭文件
file = open("data.txt")
data = file.read()
# 如果这里出错，file不会关闭
process(data)
file.close()

# ✅ 使用上下文管理器
with open("data.txt") as file:
    data = file.read()
    process(data)
# 自动关闭文件，即使出错
```

---

## 💡 最佳实践

### 1. 尽早检查，快速失败

```python
def transfer_money(from_account, to_account, amount):
    # ✅ 先检查所有条件
    if amount <= 0:
        raise ValueError("金额必须大于0")
    if from_account.balance < amount:
        raise ValueError("余额不足")
    if to_account is None:
        raise ValueError("目标账户不存在")

    # 所有检查通过，执行操作
    from_account.balance -= amount
    to_account.balance += amount
```

### 2. 具体的异常类型

```python
# ❌ 太笼统
except Exception:
    print("出错了")

# ✅ 具体处理
except FileNotFoundError:
    print("文件不存在")
except PermissionError:
    print("没有权限")
except ValueError as e:
    print(f"数据错误: {e}")
```

### 3. 不要吞掉异常

```python
# ❌ 吞掉异常
try:
    process_data()
except:
    pass  # 什么都不做，错误被隐藏了

# ✅ 至少记录日志
import logging

try:
    process_data()
except Exception as e:
    logging.error(f"处理数据失败: {e}")
    raise  # 重新抛出异常
```

### 4. 清理资源

```python
# ✅ 方式1：try-finally
file = open("data.txt")
try:
    data = file.read()
    process(data)
finally:
    file.close()  # 确保关闭

# ✅ 方式2：上下文管理器（推荐）
with open("data.txt") as file:
    data = file.read()
    process(data)
```

### 5. 自定义上下文管理器

```python
class DatabaseConnection:
    def __enter__(self):
        print("连接数据库")
        self.conn = connect_to_db()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("关闭数据库连接")
        self.conn.close()
        # 返回False表示不抑制异常
        return False

# 使用
with DatabaseConnection() as db:
    db.query("SELECT * FROM users")
# 自动关闭连接
```

---

## 🎯 错误处理策略

### 1. 异常传播

```python
def low_level():
    raise ValueError("低层错误")

def mid_level():
    low_level()  # 不处理，让错误向上传播

def high_level():
    try:
        mid_level()
    except ValueError as e:
        print(f"在高层处理: {e}")

high_level()
```

### 2. 异常转换

```python
class APIError(Exception):
    pass

def fetch_data():
    try:
        # 底层HTTP请求
        response = requests.get(url)
    except requests.ConnectionError:
        # 转换为业务异常
        raise APIError("无法连接到服务器")
    except requests.Timeout:
        raise APIError("请求超时")
```

### 3. 重试机制

```python
import time

def retry(max_attempts=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise  # 最后一次重试失败，抛出异常
                    print(f"尝试 {attempt + 1} 失败，重试...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3)
def fetch_data():
    # 可能失败的网络请求
    pass
```

### 4. 降级策略

```python
def get_user_info(user_id):
    try:
        # 尝试从数据库获取
        return db.get_user(user_id)
    except DatabaseError:
        try:
            # 降级：从缓存获取
            return cache.get_user(user_id)
        except CacheError:
            # 再降级：返回默认值
            return {"id": user_id, "name": "访客"}
```

---

## 🔗 相关概念

- [类型系统](type-systems.md) - 类型可以表达错误（Option, Result）
- [并发编程](concurrency-parallelism.md) - 并发中的错误处理更复杂
- [抽象与封装](abstraction-encapsulation.md) - 错误是接口的一部分
- [函数式编程](programming-paradigms.md) - 函数式的错误处理方式

---

## 📚 深入学习

- **书籍**：《代码大全》、《Effective Error Handling》
- **模式**：异常处理模式、容错设计
- **语言**：Rust（Result/Option）、Haskell（Maybe/Either）

---

**记住**：
1. 错误是正常的，要优雅处理
2. 尽早检查，快速失败
3. 使用具体的异常类型
4. 不要吞掉异常
5. 确保清理资源
6. 考虑重试和降级
