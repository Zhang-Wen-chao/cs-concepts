# 装饰器：@语法糖、带参数装饰器、functools.wraps、类装饰器

> 装饰器是 Python 最优雅的特性之一——本质是"传入函数，返回函数"的调用able。

## 基础：装饰器就是高阶函数

```python
# 装饰器本质
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"调用: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"返回: {result}")
        return result
    return wrapper

# 等价写法
def add(a, b):
    return a + b

add = logger(add)   # 手动装饰

# 语法糖
@logger
def add(a, b):
    return a + b

print(add(3, 4))
# 调用: add
# 返回: 7
```

## `@functools.wraps`：保留原函数元数据

```python
import functools

def logger(func):
    @functools.wraps(func)  # 复制 __name__, __doc__, __module__ 等
    def wrapper(*args, **kwargs):
        print(f"调用: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    """返回两数之和"""
    return a + b

print(add.__name__)  # "add"（没有 wraps 的话是 "wrapper"）
print(add.__doc__)   # "返回两数之和"
```

**永远用 `@functools.wraps`**，否则调试和 introspection 会疯掉。

## 带参数的装饰器

```python
import functools

def repeat(n=1):
    """装饰器工厂：多执行几次"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

# 无参数（默认 n=1）
@repeat
def shout(s): print(s.upper())

# 带参数
@repeat(n=3)
def whisper(s): print(s.lower())

shout("hi")    # HI
whisper("BYE") # bye bye bye
```

**三层嵌套的理解**：`@repeat(n=3)` → 调用 `repeat(3)` 返回 `decorator` → `@decorator` 装饰函数。

## 类装饰器

```python
import functools

class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"调用 #{self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def hello():
    print("hello!")

hello()  # 调用 #1, hello!
hello()  # 调用 #2, hello!
print(hello.count)  # 2
```

## 装饰器在实际项目中的应用

```python
import functools, time

# 1. 计时
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

# 2. 缓存（LRU）
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)

# 3. 权限检查
def require_auth(func):
    @functools.wraps(func)
    def wrapper(user, *args, **kwargs):
        if not user.get("authenticated"):
            raise PermissionError("未认证")
        return func(user, *args, **kwargs)
    return wrapper

# 4. 重试
def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(2 ** attempt)  # 指数退避
            return wrapper
        return decorator
```

## 总结

- 装饰器 = 高阶函数，`@decorator` 等价于 `func = decorator(func)`
- 永远加 `@functools.wraps` 保留元数据
- 带参数装饰器需要三层嵌套：参数层 → 装饰器层 → wrapper 层
- 类装饰器用 `__call__` 实现，可以带状态
- 装饰器在框架、日志、缓存、权限验证中无处不在
