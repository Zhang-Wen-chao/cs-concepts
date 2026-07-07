# Coroutine 底层原理：generator-based coroutine、`yield from`、`await` 的本质

> Python 的 `async/await` 是建立在 generator 之上的语法糖。理解底层原理，才能真正理解事件循环。

## 从 Generator 到 Coroutine

```python
def gen():
    yield 1
    yield 2

g = gen()
next(g)  # 1
next(g)  # 2
next(g)  # StopIteration
```

Generator 可以暂停和恢复——这本质上就是 coroutine。

## `send()`：让 generator 变成双向通道

```python
def echo():
    while True:
        received = yield
        print(f"got: {received}")

g = echo()
next(g)        # 启动到第一个 yield
g.send("hi")   # got: hi
g.send("bye")  # got: bye
```

`send()` 把值发送给 `yield` 表达式，赋值给 `received`。**这就是 coroutine 的消息传递机制。**

## `yield from`：委托给子生成器

```python
def sub_gen():
    x = yield "sub: ready"
    yield f"sub: got {x}"

def main_gen():
    result = yield from sub_gen()
    yield f"main: {result}"

g = main_gen()
next(g)               # "sub: ready"
g.send("hello")       # "sub: got hello"  — 注意：send 穿透到了 sub_gen
next(g)               # "main: None"
```

`yield from` 做了三件事：
1. 把 `send()` / `throw()` / `close()` 全部透传给子生成器
2. 接收子生成器的 `yield` 值
3. 子生成器 `return` 的值成为 `yield from` 表达式的值

**这正是 `await` 的前身。**

## `await` 的本质

```python
# Python 3.5+
async def foo():
    result = await bar()
    return result

# ≈ 等价于 Python 3.4 的写法
@types.coroutine
def foo():
    result = yield from bar()
    return result
```

```python
# await 的底层：PEP 492
import types

@types.coroutine
def sleeper(n):
    yield from asyncio.sleep(n)  # sleep 内部 yield 了 Future

# async def 只是语法糖
async def demo():
    await sleeper(1)  # await 等价于 yield from
```

关键区别：
| | Generator-based coroutine | Native coroutine (`async def`) |
|---|---|---|
| 类型 | generator | coroutine |
| await 语法 | `yield from` 隐式 | `await` 显式 |
| 谁可 await | generator / awaitable | 只接受 awaitable |
| 可迭代？ | 可迭代（易误用） | 不可迭代（安全） |

## Awaitable 协议

```python
# 一个对象可 await 当且仅当：
# 1. 是 coroutine（async def 返回值）
# 2. 实现了 __await__() 返回 iterator

class MyAwaitable:
    def __await__(self):
        yield from ready.wait()
        return result

async def use():
    result = await MyAwaitable()  # 合法
```

## 手写事件循环（简化版）

```python
def run(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

# 真正的 asyncio.run() ≈
def asyncio_run(main_coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(main_coro)
    finally:
        loop.close()
```

完整事件循环的核心逻辑：

```python
class SimpleLoop:
    def __init__(self):
        self._ready = []       # 准备好的回调
        self._timers = []      # 定时器

    def call_soon(self, callback):
        self._ready.append(callback)

    def run_once(self):
        while self._ready:
            cb = self._ready.pop(0)
            cb()

    def run_forever(self):
        while True:
            self.run_once()
            # 实际还有：epoll/select 等待 I/O 就绪
```

## Coroutine 的生命周期

```python
async def example():
    print("start")
    await asyncio.sleep(1)
    print("end")

coro = example()  # 1. 创建 coroutine 对象（尚未执行任何代码）
coro.send(None)   # 2. 执行到第一个 await → 遇到 yield → 暂停
# 3. Future 完成后 → loop 调用 coro.send(None) → 恢复执行
coro.close()      # 4. 提前关闭 → GeneratorExit 异常
```

## 与 Go goroutine 的深层对比

| | Python async/await | Go goroutine |
|---|---|---|
| 调度 | 协作式（显式 await 让出） | 抢占式（任何函数调用都可能是调度点） |
| 栈大小 | 无独立栈（复用调用栈） | 2KB 起步，可增长 |
| 创建成本 | 创建 coroutine ≈ 函数调用 | goroutine ≈ 4KB，数百万不压力 |
| 切换 | await 即切换 | 任何 syscall / chan / GC 都可能 |
| 并行 | 单线程（GIL） | M:N 调度到多线程 |
| 适用 | I/O 密集型 | I/O 和 CPU 密集型 |
