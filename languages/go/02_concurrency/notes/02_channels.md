# 02 · Channel 模式速记

> 资料：Tour of Go「[Channel](https://go.dev/tour/concurrency/2)」「Buffered Channels」「select」，Go by Example（Timers / Tickers / Worker Pools），Effective Go Channel 章节。

## 基本语义
- `make(chan T)`：无缓冲，发送必须等待接收；适合传递同步信号。
- `make(chan T, n)`：缓冲 channel，可临时缓存 n 条消息，降低生产/消费速率差异。
- **单向 channel**：函数签名中声明 `chan<- T`（仅发送）或 `<-chan T`（仅接收）以限制误用。
- `close(ch)`：告知接收端“不会再有新值”；接收时可用 `v, ok := <-ch` 判断。

## select 模式
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
- 每个 `case` 表达式必须是 channel 操作；`default` 表示非阻塞尝试。
- `time.After` 适合快速超时控制，频繁调用时请改用 `time.NewTimer` 复用。

## 模式清单
1. **Fan-out / Fan-in**：生产者发往多个 worker（fan-out），worker 结果汇总至一个 channel（fan-in）。搭配 `errgroup` 控制生命周期。
2. **Pipeline**：多个 stage 串联 `<-chan`；关闭上一层 channel 即可让下一层退出。
3. **Semaphore**：使用缓冲 channel 当作令牌池（`tokens := make(chan struct{}, limit)`），通过 send/receive 控制并发。
4. **Ticker/Timer**：用 `time.NewTicker` 驱动定时任务；记得 `defer ticker.Stop()`。

## 典型 Bug
- 向已关闭 channel 发送会 panic；因此由生产者负责 close，消费者不要 close 自己没创建的 channel。
- range channel 前一定要 close，否则 goroutine 会永久阻塞。
- 使用 `select` + `default` 实现非阻塞发送时，记得记录 drop/重试次数。

## Checklist
- [ ] 写出 fan-out/fan-in、pipeline、semaphore 模板，并能解释适用场景。
- [ ] 使用 `context` + `select` 构建超时/取消逻辑。
- [ ] 每个 channel 的 close 责任人明确，避免 double close。
