# Go 并发模型

## G-M-P 调度
- **G (goroutine)**：用户态协程，栈 ~2KB 按需扩缩，适合高并发 I/O
- **M (machine)**：OS 线程，承载 G 执行
- **P (processor)**：逻辑处理器，`GOMAXPROCS`（默认=CPU核数）决定并发度
- 调度器在 P 的 run queue 中调度 G→M，用户不直接管线程

## goroutine 生命周期
- 启动：`go funcName()`
- **任何 goroutine 必须有退出路径**：`<-ctx.Done()` / close channel / 超时
- 泄漏检测：`runtime.NumGoroutine()` / pprof goroutine dump

## Channel
- `make(chan T)` — 无缓冲，同步传递
- `make(chan T, n)` — 有缓冲，可容忍临时速率差异
- 单向 channel：函数签名中 `chan<- T` / `<-chan T` 限制误用
- 由**生产者**负责 close，消费者不要 close 非自己创建的 channel
- `select` 多路复用 channel + context 取消 + 超时

```go
select {
case v := <-data:
    handle(v)
case <-ctx.Done():
    return ctx.Err()
case <-time.After(2 * time.Second):
    return errors.New("timeout")
}
```

## 常用并发模式

| 模式 | 实现 | 适用场景 |
|------|------|----------|
| Fan-out/Fan-in | 多个 worker 消费 + 汇总 channel | 并发处理任务 |
| Pipeline | 多 stage 串联 `<-chan`，关闭上游即可退出 | 数据处理链 |
| Semaphore | 缓冲 channel 作令牌池 `make(chan struct{}, limit)` | 限制并发数 |
| Worker Pool | errgroup + channel + context | 可控并发 + 错误传播 |

## sync 工具

| 工具 | 用途 | 注意 |
|------|------|------|
| `sync.Mutex` | 独占锁 | 配合 `defer mu.Unlock()` |
| `sync.RWMutex` | 读写锁 | 读多写少时提升吞吐 |
| `sync.WaitGroup` | 等待 goroutine 完成 | Add/Done 严格配对，不可复制 |
| `sync.Once` | 只执行一次初始化 | 懒加载 |
| `sync.Map` | 读多写少/不可预测 key 的 map | 不替代标准 map |
| `sync/atomic` | lock-free 计数器/状态 | 理解 happens-before |

## Context 取消与超时
- `context.Background()` — 根节点，main/test 使用
- `context.WithCancel / WithTimeout / WithDeadline` — 创建可取消的子 context
- 签名约定：`func Run(ctx context.Context, ...) error` — ctx 为第一参数
- 创建后立即 `defer cancel()` 防泄漏
- 长循环/阻塞操作必须监听 `<-ctx.Done()`

## errgroup
```go
g, ctx := errgroup.WithContext(ctx)
g.Go(func() error { return process(ctx, task) })
if err := g.Wait(); err != nil { return err }
```
- 自动捕获首个错误并取消 context
- 如需限制并发：叠加信号量 `semaphore.NewWeighted(limit)`

## 限流
- `semaphore.NewWeighted(n)` — 信号量控制并发数
- `rate.NewLimiter(r, burst)` — 令牌桶，适合 API QPS 控制
- 重试：指数退避 + 抖动；永久错误（4xx）直接返回

## playground
- [`02_concurrency/playground`](./02_concurrency/playground/) — 爬虫、context 超时、sync 限流、errgroup pipeline
