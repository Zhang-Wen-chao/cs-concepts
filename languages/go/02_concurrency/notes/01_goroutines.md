# 01 · Goroutine 调度总览

> 资料：Tour of Go「Concurrency」→ [Goroutines](https://go.dev/tour/concurrency/1)，Go Blog「[Go Schedulers](https://go.dev/blog/scheduler)」，`runtime` 包文档。

## 核心概念
- **G（goroutine）**：用户态协程，栈默认 ~2KB，按需增长；非常适合高并发 I/O。
- **M（machine）**：绑定操作系统线程，负责承载多个 G。
- **P（processor）**：逻辑处理器，决定同时运行多少 G；数量由 `GOMAXPROCS` 控制。
- 调度器负责在 P 的 run queue 中调度 G → M，避免用户手工管理线程。

## 常用运行时工具
- `runtime.GOMAXPROCS(n)`：限制同时执行 goroutine 的最大数量，默认=CPU 核心。
- `runtime.NumGoroutine()`：观测当前 goroutine 数量，定位泄漏。
- `runtime.Gosched()`：当前 G 主动让出时间片（仅在 demo/练习中使用，生产代码更依赖 channel 同步）。
- `debug.SetGCPercent()`：高并发 + 内存敏感时调整 GC 策略。

## 模式示例
```go
func spawnWorkers(ctx context.Context, n int, jobs <-chan Job) <-chan Result {
    results := make(chan Result)
    var wg sync.WaitGroup
    wg.Add(n)

    for i := 0; i < n; i++ {
        go func(id int) {
            defer wg.Done()
            for job := range jobs {
                select {
                case <-ctx.Done():
                    return
                case results <- job.Do(id):
                }
            }
        }(i)
    }

    go func() {
        wg.Wait()
        close(results)
    }()
    return results
}
```
- 通过 `ctx.Done()` 终止 goroutine，防止泄漏。
- 生产者消费 channel 后立即 `close`，消费端使用 `for range` 自动退出。

## 实战建议
1. 任何 goroutine 必须有退出路径（context/close channel/超时）。
2. 用 `errgroup.WithContext` 管理 goroutine 树形依赖；避免到处 `sync.WaitGroup`。
3. 观察点：`pprof` 中 goroutine 数量曲线是否与请求量成比例；如不断增加需排查泄漏。
4. `GOMAXPROCS` 只有在 CPU 密集型任务下需要调整；I/O 型多数情况保持默认即可。

## Checklist
- [ ] 能口述 G-M-P 模型、`GOMAXPROCS` 默认值、何时需要修改。
- [ ] 会将 goroutine 封装进函数，保证外部只感知 channel/context。
- [ ] 知道如何借助 `runtime/pprof` 采集 goroutine dump，并定位阻塞链路。
