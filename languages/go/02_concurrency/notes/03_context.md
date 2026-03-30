# 03 · Context 取消与超时

> 资料：Go Blog「[Context](https://go.dev/blog/context)」，`context` 包文档，《Go Programming Language》第 8 章。

## 树状传播
- `context.Background()`：根节点，常用于 main/test。
- `context.WithCancel(parent)`：返回子 context + cancel 函数；调用 cancel 会向子树广播。
- `context.WithTimeout(parent, d)` / `WithDeadline`：自动在 d 之后 cancel，不调用 cancel 会泄漏定时器。
- `context.WithValue(parent, key, value)`：仅存跨 API 的请求范围数据，禁止传业务数据/大对象。

## 使用准则
1. **签名放首位**：`func Run(ctx context.Context, ...) error`，方便向下传递。
2. **传不可变值**：Value key 必须是自定义类型，避免冲突；value 不要存指针修改。
3. **立即 defer cancel**：创建 context 后立即 `defer cancel()`，防止资源泄漏。
4. **尊重 Done**：长循环/阻塞操作必须同时监听 `<-ctx.Done()`。

## 示例
```go
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
resp, err := http.DefaultClient.Do(req)
```
- HTTP 请求会在 context 超时时自动取消。
- Worker pool 中也要传 ctx，使 goroutine 能被及时回收。

## Checklist
- [ ] 任何导出函数接收 `context.Context`，且第一参数。
- [ ] 明白何时需要 `WithValue`（traceID、请求 metadata），何时改用函数参数。
- [ ] 会抓取 `ctx.Err()`（`context.DeadlineExceeded` / `context.Canceled`）并据此判定是否重试。
