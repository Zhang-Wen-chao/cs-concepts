# async/await：事件循环、coroutine、Task/Future、与 Go goroutine 对比

## 为什么需要 async/await

```python
# 同步：一个一个来
def fetch_sync(urls):
    import time, requests
    start = time.time()
    results = []
    for url in urls:
        results.append(requests.get(url))
        print(f"{url} done")
    print(f"total: {time.time()-start:.2f}s")

# async：可以并发（I/O 密集）
async def fetch_async(urls):
    import asyncio, aiohttp
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        for task in asyncio.as_completed(tasks):
            resp = await task
            print(f"{resp.url} done")
    print(f"total: {time.time()-start:.2f}s")
```

**本质**：async/await 让 I/O 等待时不阻塞事件循环，而是让出控制权去处理其他任务。

## 核心组件

```python
import asyncio

# 1. Coroutine（协程函数）
async def hello():
    return "hello"

# 2. await：挂起当前协程，等另一个协程完成
async def greet():
    result = await hello()  # 挂起，不阻塞事件循环
    print(result)

# 3. Task：将协程包装为可调度的任务
async def main():
    task = asyncio.create_task(hello())  # 创建任务
    result = await task                   # 等任务完成
    print(result)

# 4. 运行入口
asyncio.run(main())
```

## 事件循环的工作方式

```python
import asyncio

async def task(name, delay):
    print(f"{name} 开始")
    await asyncio.sleep(delay)  # 模拟 I/O
    print(f"{name} 结束")
    return f"{name} 结果"

async def main():
    # 并发运行多个任务
    results = await asyncio.gather(
        task("任务A", 2),
        task("任务B", 1),
        task("任务C", 3),
    )
    print(results)  # ['任务A 结果', '任务B 结果', '任务C 结果']

asyncio.run(main())
# 任务A 开始
# 任务B 开始
# 任务C 开始
# 任务B 结束（1秒后）
# 任务A 结束（2秒后）
# 任务C 结束（3秒后）
```

**事件循环是单线程**：同一个时间点只有一个任务在执行，`await` 的时候切换到另一个任务。

## Future 和 Task

```python
# Future：未来某个时刻的结果（低层接口）
# Task：Future 的子类，包装 coroutine

async def slow_op():
    await asyncio.sleep(1)
    return 42

async def main():
    loop = asyncio.get_running_loop()
    
    # 显式创建 Future
    fut = loop.create_future()
    loop.call_soon(lambda: fut.set_result(42))
    print(await fut)  # 42
    
    # Task 更常用
    task = asyncio.create_task(slow_op())
    # 可以取消
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("取消了")
```

## asyncio 的实用模式

```python
import asyncio, aiohttp

# 1. 限制并发数
sem = asyncio.Semaphore(10)

async def fetch(session, url):
    async with sem:
        async with session.get(url) as resp:
            return await resp.text()

# 2. 超时控制
try:
    async with asyncio.timeout(5):
        result = await slow_operation()
except TimeoutError:
    print("超时")

# 3. 同步原语
lock = asyncio.Lock()     # 互斥锁
event = asyncio.Event()   # 事件
queue = asyncio.Queue()   # 队列（生产者消费者）
```

## 与 Go goroutine 对比

| 特性 | Python asyncio | Go goroutine |
|---|---|---|
| 调度 | 单线程事件循环（协作式） | M:N 调度（抢占式） |
| 并发模型 | async/await | goroutine + channel |
| 切换点 | `await` 处显式挂起 | 任何函数调用都可能切换 |
| 并行 | 不并行（单线程） | 可并行（多核） |
| 内存 | 默认小（~几 KB） | 极小（~2 KB 起始） |
| CPU 密集 | 不适用（GIL 卡死） | 适用（自动利用多核） |
| 学习曲线 | 需要理解事件循环 | 更直觉 |
| 标准库 | asyncio + aio* 生态 | 原生支持网络、文件 I/O |

```python
# Python：显式 await
async def main():
    await fetch("http://...")
    await write_db(...)
asyncio.run(main())
```

```go
// Go：不用显式等待，go 关键字就调度
func main() {
    go fetch("http://...")
    go writeDb(...)
    // waitGroup 或 channel 同步
}
```

**一句话**：Python asyncio 是**协作式单线程并发**，Go goroutine 是**抢占式 M:N 并发**。asyncio 适合 I/O 密集型，不适合 CPU 密集型。Go goroutine 两者都适合。

## 总结

- async/await 是 Python 生成器协程的语法升级（`yield` → `await`）
- 事件循环是单线程的，`await` 时切到其他协程
- `asyncio.gather`、`Semaphore`、`timeout` 是常用工具
- Python asyncio vs Go goroutine：协作式 vs 抢占式，单线程 vs 多核
- Python 的 async 不解决 CPU 并行问题（那是 multiprocessing 的事）
