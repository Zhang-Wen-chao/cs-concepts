# 迭代器与生成器：\_\_iter\_\_/\_\_next\_\_、yield、yield from、协程基础

## 迭代器协议

```python
# 任何实现了 __iter__ 和 __next__ 的对象都是迭代器
class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0
    
    def __iter__(self):
        return self  # 迭代器返回自身
    
    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration  # 终结信号
        self.current += 1
        return self.current - 1

for n in Counter(5):
    print(n)  # 0, 1, 2, 3, 4
```

**协议关系**：
- 可迭代（iterable）：实现了 `__iter__`，返回一个迭代器
- 迭代器（iterator）：实现了 `__iter__` + `__next__`

```python
# Python 内部是这样遍历的：
it = iter([1, 2, 3])   # 调用 __iter__
while True:
    try:
        value = next(it)  # 调用 __next__
        print(value)
    except StopIteration:
        break
```

## 生成器：用 yield 替代 return

```python
def count_up_to(limit):
    n = 0
    while n < limit:
        yield n  # 暂停，返回 n
        n += 1   # 下次从这里继续

gen = count_up_to(5)
print(next(gen))  # 0
print(next(gen))  # 1
print(list(gen))  # [2, 3, 4] — 继续消费剩下的
```

**生成器是迭代器**，但更简洁。每次 `yield` 暂停函数、记住状态，下次 `next()` 从暂停处继续。

## 生成器表达式

```python
# 列表推导
squares_list = [x*x for x in range(10)]    # 一次性全创建

# 生成器表达式
squares_gen = (x*x for x in range(10))     # 惰性求值

print(type(squares_gen))   # <class 'generator'>
print(sum(squares_gen))    # 285 — 逐个计算，不创建大列表
```

**什么时候用生成器**：处理大量数据时节省内存，或无限序列。

## `yield from`：委托给子生成器

```python
def chain(*iterables):
    for it in iterables:
        yield from it  # 等价于 for x in it: yield x

print(list(chain("AB", [1, 2], {"x": 1})))  # ['A', 'B', 1, 2, 'x']
```

`yield from` 会把子生成器的所有值逐个 yield，不用手动写循环。

## 协程基础：send、throw、close

```python
def echo():
    """用 yield 同时接收和发送值"""
    print("开始")
    try:
        while True:
            received = yield  # yield 表达式接收值
            print(f"收到: {received}")
    except GeneratorExit:
        print("生成器关闭")

gen = echo()
next(gen)        # 启动到第一个 yield（必须）
gen.send("hi")   # 收到: hi
gen.send(42)     # 收到: 42
gen.close()      # 生成器关闭
```

```python
# 协程用途：累加器
def accumulator():
    total = 0
    while True:
        value = yield total
        total += value

acc = accumulator()
next(acc)              # 启动
print(acc.send(10))    # 10
print(acc.send(5))     # 15
print(acc.send(3))     # 18
```

**生成器状态**：GEN_CREATED → GEN_RUNNING → GEN_SUSPENDED → GEN_CLOSED

## 实战：流水线处理

```python
def read_logs(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip()

def filter_errors(lines):
    for line in lines:
        if "ERROR" in line:
            yield line

def extract_timestamps(lines):
    import re
    for line in lines:
        match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', line)
        if match:
            yield match.group()

# 组合成流水线（没有中间大列表！）
pipeline = extract_timestamps(filter_errors(read_logs("app.log")))
for ts in pipeline:
    print(ts)
```

## 总结

| 概念 | 说明 |
|---|---|
| 迭代器 | 实现 `__iter__` + `__next__`，支持 for 循环 |
| 生成器 | 用 `yield` 的迭代器，惰性求值，暂停/恢复 |
| `yield from` | 委托给子生成器，简化嵌套生成器 |
| 生成器表达式 | `(x for x in ...)` 惰性版本 |
| send | 向生成器发送值，实现双向通信（协程基础） |
| close | 关闭生成器，触发 GeneratorExit |

**对比 Go**：Python 生成器 ≈ Go 的 channel + goroutine 的轻量级版本。Python 用 `yield` 暂停函数，Go goroutine 是 M:N 调度。Python 协程后来升级为 `async def`（见下一篇）。
