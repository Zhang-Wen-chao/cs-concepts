# 上下文管理器：\_\_enter\_\_/\_\_exit\_\_、@contextmanager、与 C++ RAII 对比

> Python 的 with 语句 ≈ C++ 的 RAII。但实现不一样。

## 手动实现上下文管理器

```python
class ManagedFile:
    def __init__(self, name, mode='r'):
        self.name = name
        self.mode = mode
    
    def __enter__(self):
        self.file = open(self.name, self.mode)
        return self.file  # as 子句拿到的东西
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # 返回 True 会抑制异常
        return False

with ManagedFile('test.txt', 'w') as f:
    f.write('hello')
# 离开 with 块自动调用 __exit__
```

**`__exit__` 的三个参数**：
- `exc_type`：异常类型（没异常是 None）
- `exc_val`：异常实例
- `exc_tb`：traceback

## @contextmanager 装饰器

```python
from contextlib import contextmanager

@contextmanager
def managed_file(name, mode='r'):
    # __enter__
    f = open(name, mode)
    try:
        yield f  # 这就是 as 拿到的
    finally:
        # __exit__
        f.close()

with managed_file('test.txt', 'w') as f:
    f.write('hello')
```

**用 `@contextmanager` 的规则**：
- `yield` 之前的代码是 `__enter__`
- `yield` 之后的代码是 `__exit__`（即使异常也会执行）
- 整个 `yield` 包在 `try/finally` 里最安全

## contextlib 工具集

```python
import contextlib

# 1. closing：自动调用 .close()
from contextlib import closing
with closing(open('test.txt')) as f:
    pass  # 自动 f.close()

# 2. suppress：忽略指定异常
with contextlib.suppress(FileNotFoundError):
    os.remove('temp.txt')

# 3. redirect_stdout：临时重定向
import io
with contextlib.redirect_stdout(io.StringIO()) as buf:
    print("这段输出不会打印")
    print("而是被捕获了")
print(buf.getvalue())  # "这段输出不会打印\n而是被捕获了"

# 4. ExitStack：动态管理多个上下文管理器
filenames = ['a.txt', 'b.txt', 'c.txt']
with contextlib.ExitStack() as stack:
    files = [stack.enter_context(open(f)) for f in filenames]
    # 退出时全部自动 close
```

## 与 C++ RAII 对比

```python
# Python：with 语句保证资源释放
with open('f.txt') as f:
    data = f.read()
```

```cpp
// C++：RAII 在构造函数获取，析构函数释放
std::ifstream f("f.txt");
std::string data;
f >> data;
// 离开作用域自动析构释放
```

| 方面 | Python | C++ |
|---|---|---|
| 机制 | `__enter__` / `__exit__` 协议 | 构造函数/析构函数 |
| 触发 | 显式 `with` 语句 | 作用域自动触发 |
| 异常安全 | `__exit__` 总被执行 | 析构函数总被执行 |
| 生命周期 | `with` 块内 | 对象作用域内 |
| 嵌套 | `ExitStack` | 作用域嵌套天然支持 |

**Python 没有真正的 RAII**：因为 GC 和引用计数不可预测。`__del__` 不是析构函数——它由 GC 触发，时机不确定。所以 Python 用 `with` 语句显式管理生命周期。

```python
# ⚠️ 不要依赖 __del__ 释放资源！
class Resource:
    def __del__(self):
        print("清理资源")  # 不确定什么时候执行

# ✅ 用上下文管理器
class Resource:
    def __enter__(self):
        print("获取资源")
        return self
    def __exit__(self, *args):
        print("释放资源")
```

## 实际应用

```python
# 数据库事务
class Transaction:
    def __init__(self, db):
        self.db = db
    
    def __enter__(self):
        self.db.begin()
        return self
    
    def __exit__(self, exc_type, *args):
        if exc_type is None:
            self.db.commit()
        else:
            self.db.rollback()
        return False  # 不抑制异常

# 计时器
@contextmanager
def timer(label="block"):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter()-start:.3f}s")
```

## 总结

- 上下文管理器 = 资源获取 ‍→ 使用 → 资源释放 的协议
- 两种实现：类（`__enter__`/`__exit__`）或 `@contextmanager` 装饰器
- contextlib 提供了 `suppress`、`closing`、`ExitStack` 等实用工具
- Python 的 with ≈ C++ 的 RAII，但 Python 需要显式 `with`，C++ 靠析构自动触发
- 永远用上下文管理器管理资源，不用 `__del__`
