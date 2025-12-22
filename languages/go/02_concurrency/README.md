# 阶段 2 · 并发

> 目标：熟练使用 goroutine、channel、context、sync、errgroup 等原语，并完成一个受限并发的爬虫 `playground/01_crawler/cmd/crawler`。

## 学习闭环
| 步骤 | 资料 | 产出 |
| --- | --- | --- |
| 阅读 | `notes/01_goroutines.md` ~ `05_errgroup_rate_limiting.md` + Go Tour Concurrency / Go Blog | 笔记 + 疑问列表 |
| 实验 | `playground/01_crawler/internal/crawler` | 逐步实现 worker pool / context / 限流 |
| 运行 | `go fmt ./... && go test ./... && go test -race ./...` | 通过 race detector + table tests |
| 记录 | `go_cheatsheet.md` 并发篇 | 追加 goroutine/channel/context 速查 |

## Checklist
- [ ] 明确 Go scheduler、`GOMAXPROCS`、`runtime.Gosched` 的作用与限制。
- [ ] 熟能生巧：无缓冲 vs 缓冲 channel，fan-out/fan-in、select + timeout 模式写成代码片段。
- [ ] context 生命周期清晰，知道何时 `WithCancel`/`WithTimeout`，禁止滥用 `context.Value`。
- [ ] `sync.Mutex/RWMutex/WaitGroup/Once` + `atomic.Value` 的适用场景都能列举并写出模板。
- [ ] errgroup + semaphore 限流示例跑通，并记录默认并发度策略。
- [ ] `crawler` 支持：worker pool（<=20 并发）、context 超时、失败重试、统计响应时间/状态码。
- [ ] `crawler` 单元测试：使用 `httptest.Server` 模拟延迟/失败，并通过 `go test -race ./...`。

## Playground 模块
- `01_crawler/...` —— worker pool + context + 限流 CLI。
- `02_context_guard` —— `RunWithTimeout` 演示 `context.WithTimeout` 的包装用法。
- `03_sync_limiter` —— channel 信号量实现，保证最大并发。
- `04_errgroup_pipeline` —— `errgroup + semaphore` 处理批量任务并保持顺序。

## 验收方式
```bash
cd languages/go/02_concurrency/playground
go fmt ./...
go test ./...
go test -race ./...
go run 01_crawler/cmd/crawler --urls 01_crawler/fixtures/urls.txt --timeout 3s --max-workers 20
```
- 运行时打印 JSON/表格统计，且超时/失败有明确日志。
- `urls.txt` 可放在 `playground/01_crawler/fixtures`，包含慢速与失败 URL 以验证重试逻辑。

## 复盘要点
- 哪些逻辑应该用 channel，哪些更适合 `sync`？写下选择理由。
- 在 race detector 下暴露了哪些数据竞争？如何修复。
- errgroup/限流策略是否可复用到 Stage 4 的综合项目。
