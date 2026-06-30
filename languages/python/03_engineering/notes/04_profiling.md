# 性能分析：cProfile、timeit、line_profiler、优化技术

> Python 以慢闻名——但你知道慢在哪之前，别优化。测量，再优化。

## timeit：微基准测试

```python
import timeit

# 测试小段代码
t = timeit.timeit(
    '"-".join(str(n) for n in range(100))',
    number=10000
)
print(f"10,000 次耗时: {t:.3f}s")

# 更精确：重复多次取最小
min_t = min(timeit.repeat(
    setup='import math',  # 准备环境
    stmt='math.sqrt(144)',
    repeat=5,
    number=100000
))
print(f"最佳: {min_t:.6f}s")

# 命令行
# python -m timeit "'-'.join(str(n) for n in range(100))"
```

## cProfile：函数级性能分析

```python
import cProfile, pstats

def slow_func():
    total = 0
    for i in range(100_000):
        total += i ** 2          # 平方运算
    return total

def fast_func():
    total = 0
    for i in range(100_000):
        total += i * i           # 乘法比平方快
    return total

# 用 cProfile 分析
profiler = cProfile.Profile()
profiler.enable()
slow_func()
fast_func()
profiler.disable()

# 排序输出
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)  # 前 10 个最耗时函数
```

```bash
# 命令行方式更好用
python -m cProfile -o profile.prof my_script.py

# 可视化（用 snakeviz）
pip install snakeviz
snakeviz profile.prof    # 浏览器打开交互式火焰图
```

## line_profiler：逐行分析

```python
# pip install line_profiler
@profile  # 只装饰要分析的函数
def process_data(items):
    result = []
    for item in items:
        temp = item * 2
        temp = temp ** 0.5
        result.append(temp)
    return result

process_data(list(range(10_000)))

# 运行：kernprof -l -v my_script.py
# 输出每行耗时、次数、百分比
```

**line_profiler 比 cProfile 更精细**——能精确到每行代码。但需要 `@profile` 装饰器。

## memory_profiler：内存分析

```python
# pip install memory_profiler
@profile
def create_list(n):
    result = []
    for i in range(n):
        result.append(list(range(100)))
    return result

create_list(1000)

# 运行：python -m memory_profiler my_script.py
# 输出每行内存增量
```

```python
# 实时内存监控
from memory_profiler import memory_usage

def my_func():
    data = [i for i in range(10_000_000)]
    return sum(data)

usage = memory_usage((my_func, (), {}), interval=0.1)
print(f"峰值: {max(usage):.1f} MB")
```

## `__slots__` 优化

```python
# 大量实例时 __slots__ 效果显著

# 不用 __slots__
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 用 __slots__
class SlotPoint:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys
p1 = Point(1, 2)
p2 = SlotPoint(1, 2)
print(f"Point: {sys.getsizeof(p1)} bytes")      # ~56 bytes
print(f"SlotPoint: {sys.getsizeof(p2)} bytes")  # ~48 bytes
# 实际上节省更多，因为没有了 __dict__（~数百字节）
```

## C 扩展：ctypes 与 Cython

```python
# 1. ctypes: 直接调用 C 库
import ctypes

# 加载系统数学库
libm = ctypes.CDLL("libm.dylib")  # macOS
# libm = ctypes.CDLL("libm.so.6")  # Linux

libm.sqrt.argtypes = [ctypes.c_double]
libm.sqrt.restype = ctypes.c_double
print(libm.sqrt(42.0))  # 6.48...
```

```cython
# 2. Cython: Python 语法编译为 C
# calc.pyx
def sum_of_squares(int n):
    cdef int i, total = 0
    for i in range(n):
        total += i * i
    return total
```

```python
# 3. 简单优化：用内置函数替代手写循环
# ❌ 慢
total = 0
for i in range(100_000):
    total += i

# ✅ 快（C 层实现）
total = sum(range(100_000))

# ❌ 慢
squares = []
for x in nums:
    squares.append(x * x)

# ✅ 快（列表推导优化）
squares = [x * x for x in nums]
```

## Python 优化思路总结

```python
# 优化优先级：
# 1. 算法复杂度（O(n²) → O(n log n)）— 最大收益
# 2. 减少 Python 层循环 → 多推到 C 层
# 3. 用合适的数据结构（set 查 O(1) vs list 查 O(n)）
# 4. __slots__ 减少内存
# 5. C 扩展（最后的手段）

# 常见优化：
# 局部变量比全局变量快
import math
def fast():
    sqrt = math.sqrt   # 局部引用
    for i in range(1000):
        sqrt(i)

# 字符串拼接用 join 不用 +
result = "".join(parts)       # ✅
result = ""                   # ❌
for p in parts:
    result += p
```

## 总结

| 工具 | 用途 | 命令 |
|---|---|---|
| `timeit` | 微基准测试 | `python -m timeit '代码'` |
| `cProfile` | 函数级性能分析 | `python -m cProfile script.py` |
| `snakeviz` | 可视化火焰图 | `snakeviz profile.prof` |
| `line_profiler` | 逐行分析 | `kernprof -l -v script.py` |
| `memory_profiler` | 内存分析 | `python -m memory_profiler script.py` |
| `__slots__` | 内存优化 | 类定义时加 |
| Cython/ctypes | 加速热路径 | 最后手段 |

**永远先测量再优化**。90% 的情况换算法比换实现效果好得多。
