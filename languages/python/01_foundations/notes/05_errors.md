# 错误与异常：try/except/else/finally、自定义异常、异常链、contextlib

## try/except/else/finally 完整结构

```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"值错误: {e}")
except (TypeError, RuntimeError) as e:
    print(f"类型或运行时错误: {e}")
except Exception:  # 一般不裸 except
    print("未知错误")
    raise           # 重新抛出
else:
    print(f"没有异常，结果: {result}")  # 仅 try 成功时执行
finally:
    print("无论如何都执行")           # 资源清理
```

**执行顺序**：try → 无异常 → else → finally
try → 有异常 → except → finally

**else 块用在哪**：把正常流程和异常处理分开，避免在 try 里写太多不抛异常的代码。

## 自定义异常

```python
class DatabaseError(Exception):
    """数据库操作基类异常"""
    pass

class ConnectionError(DatabaseError):
    def __init__(self, host, port, original=None):
        self.host = host
        self.port = port
        self.original = original
        super().__init__(f"不能连接到 {host}:{port}")

class QueryError(DatabaseError):
    pass

# 使用
try:
    raise ConnectionError("db.example.com", 5432)
except DatabaseError as e:
    print(f"数据库错误: {e.host}:{e.port}")
```

**最佳实践**：定义自己的异常层级（继承 Exception），让调用方可以精确捕获。

## 异常链

```python
def process_data():
    try:
        result = 1 / 0  # ZeroDivisionError
    except ZeroDivisionError as e:
        raise ValueError("数据处理失败") from e  # 保留原始异常

try:
    process_data()
except ValueError as e:
    print(e)           # 数据处理失败
    print(e.__cause__) # ZeroDivisionError: division by zero
```

```python
# 隐式异常链
def load_config():
    try:
        open("config.json")
    except FileNotFoundError as e:
        raise RuntimeError("配置加载失败") from None  # 省略原因

# from None 隐藏底层异常细节
```

## contextlib.suppress

```python
import os, contextlib

# 传统写法
try:
    os.remove("temp.txt")
except FileNotFoundError:
    pass

# 简洁写法
with contextlib.suppress(FileNotFoundError):
    os.remove("temp.txt")
```

```python
# suppress 可以处理多个异常
with contextlib.suppress(FileNotFoundError, PermissionError):
    os.remove("locked_file.txt")
```

## 与其他语言的对比

| 特性 | Python | C++ | Go |
|---|---|---|---|
| 基础方式 | `try/except` | `try/catch` | `if err != nil` |
| finally | ✅ `finally` | 无（靠 RAII） | `defer` |
| 异常类型 | 继承 Exception | 任意类型 | error interface |
| 异常链 | `from` 语法 | `std::throw_with_nested` | `fmt.Errorf("%w")` |
| 不该用异常 | 控制流 | 大部分场景 | 所有场景（Go 理念） |

**Go 特别注意**：Go 没有异常，错误是值（`error` interface），通过 `return err` 传播。Python 异常是控制流的一部分，Go 里这是 anti-pattern。

## 总结

- `else` 块在无异常时执行，让正常/异常路径分离
- 自定义异常继承 Exception，构建异常层级
- `raise ... from e` 保留异常链，`from None` 截断
- `contextlib.suppress` 替代 `try/except: pass` 的冗长写法
- Python 的异常机制和 C++/Go 哲学不同——Python 信奉"请求原谅比请求许可容易"（EAFP）
