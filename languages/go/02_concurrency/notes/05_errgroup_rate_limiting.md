# 05 · errgroup / 限流实践

> 资料：`golang.org/x/sync/errgroup`, `x/sync/semaphore`, Go Blog「Pipelines」, Stripe/Cloudflare 的 worker pool 文章。

## errgroup 快速回顾
```go
g, ctx := errgroup.WithContext(ctx)
for i := range tasks {
    task := tasks[i]
    g.Go(func() error {
        return process(ctx, task)
    })
}
if err := g.Wait(); err != nil {
    return err
}
```
- 自动捕获首个错误并取消 context。
- 默认并发量 = 循环次数，如需限制需加信号量。

## 限流策略
1. **信号量**：`sem := semaphore.NewWeighted(int64(limit))`；每次 `Acquire` / `Release` 包裹耗时操作。
2. **令牌桶**：`rate.NewLimiter(r, burst)`，适合 API QPS 控制。
3. **Sized errgroup**：为 errgroup 包装信号量，或直接写缓冲 channel 作为 worker 数限制。

## 重试策略
- 短暂错误（429/5xx）→ 指数退避 + 抖动（`time.Sleep(base * (1 << attempt))`）。
- 永久错误（4xx）→ 直接返回，让 errgroup 停止其他 worker。
- 记录 metrics + structured log，方便 Stage 3/4 观测。

## Checklist
- [ ] errgroup 任务中及时检查 `ctx.Err()`，避免获知错误后仍继续执行。
- [ ] 能实现“最大并发 N + 总超时 M + 单次重试 1”的爬虫逻辑。
- [ ] rate/semaphore 参数写入配置，避免硬编码。
