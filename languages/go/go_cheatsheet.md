# Go 核心小抄

> 按阶段记录：01 基础 → 02 并发 → 03 工程化 → 04 综合项目。

---

## 01. Go Mindset（notes/01_go_mindset.md + playground/01_mindset/greet）

**核心理念**
1. 工具先行：写任何 Go 代码前跑 `go fmt ./...`、`go test ./...`，风格与正确性交由工具兜底。
2. 声明即文档：导出函数/类型必须写 doc comment（`// Greeter ...`），否则 `go doc`/pkg.go.dev 无法生成说明。
3. Less is More：组合优于继承，遇到并发问题优先 goroutine + channel（“share memory by communicating”）。

**实践（flag + 表驱动测试）**
- `flag.String(name, default, help)` 返回 `*string`，先 `flag.Parse()` 再 `*name` 解引用。
  ```go
  name := flag.String("name", "gopher", "target")
  flag.Parse()
  fmt.Println(greeting(*name, *lang))
  ```
- `:=` 是短变量声明，只能在函数内部使用；`greeting(*name, *lang)` 返回 `(string, error)`，用 `msg, err := ...` 接收。
- 表驱动测试：`cases := []struct{ ... }{... }` + `t.Run` 循环，`t.Parallel()` 可并行执行子测试。

---

## 02. Syntax Basics（notes/02_syntax_basics.md + playground/02_syntax_basics/stats）

1. 变量/常量：`var count int` 自动赋零值；函数内可用 `:=`。常量在编译期确定，例如 `const Pi = 3.14`。
2. 多返回值：`func describeNumbers(nums []int) (Report, error)` —— 先判断 `err != nil` 再使用结果；命名返回值仅在极短函数中使用。
3. 控制流：`if/for/switch` 都支持短语句（`if v, err := ...; err != nil { ... }`），`switch` 默认 `break`。切片遍历形态 `for i, v := range nums`，可用 `_` 忽略索引。

---

## 03. Collections（notes/03_collections.md + playground/03_collections/slices）

- 切片共享底层数组，写操作前 `clone := append([]T(nil), src...)` 以免外部修改。
- `Chunk(nums, size)` 模式：预估容量 `(len(nums)+size-1)/size`，循环使用 `append([]T(nil), nums[i:end]...)` 拷贝块。
- map 是引用语义；`MergeCounters(base, delta)` 可直接累加，必要时对 `nil` map 先 `make`。

---

## 04. Structs & Interfaces（notes/04_structs_interfaces.md + playground/04_interfaces/shapes）

- 方法接收者：值接收者适合只读，指针接收者修改状态；方法集决定接口实现。
- 小接口优先：`type Shape interface { Area() float64; Perimeter() float64 }`；任意 struct 只要实现即可“鸭子类型”。
- type switch：`switch v := s.(type)` 根据具体类型生成描述/分支逻辑。

---

## 05. Error Handling（notes/05_error_handling.md + playground/05_errors/validator）

- 哨兵错误：`var ErrEmptyName = errors.New("empty name")`，通过 `errors.Is(err, ErrEmptyName)` 判断。
- `fmt.Errorf("context: %w", err)` 保留原始错误链；`errors.Join` 聚合多处校验失败。
- 约定：库函数返回错误，由调用方决定日志/重试；仅不可恢复的编程错误使用 `panic`。

---

## 06. Generics（notes/06_generics.md + playground/06_generics/transform）

- 函数类型参数：`func MapSlice[T any, R any](in []T, fn func(T) R) []R`，`any` 即任意类型。
- 约束：`comparable` 允许作为 map key，`constraints.Ordered` 适合排序/比较。
- 模板：Map/Filter/Keys 组合起来即可快速构建复合操作。

---

## 07. Concurrency Toolkit（02_concurrency/notes）

- **goroutine 调度**：`GOMAXPROCS = runtime.NumCPU()`，CPU 密集场景可调；`runtime.Gosched()` 仅在 demo 使用。
- **worker pool**：`jobs := make(chan work)` + `spawnWorkers(ctx, jobs)`；关闭 channel 通知 worker 退出。
- **select 超时模板**：`select { case <-ctx.Done(): return ctx.Err() case <-time.After(d): ... }`。
- **errgroup 限流**：`sem := semaphore.NewWeighted(limit)` + `errgroup.WithContext` 管理 goroutine 生命周期。
- **Playground**：`playground/02_context_guard`（RunWithTimeout）、`03_sync_limiter`（channel 信号量）、`04_errgroup_pipeline`（errgroup + semaphore）。

---

## 08. Tooling & Testing（03_engineering/notes）

- Go Modules：`go env GOMOD` 查模块根，`go list -m all` 查依赖，`go mod tidy` 清理。
- Lint 流程：`go fmt ./...` → `goimports -w .` → `go vet ./...` → `golangci-lint run ./...`。
- Table-driven + subtest + `t.Parallel()`；Benchmark 使用 `go test -bench Name -benchmem -run ^$` 并关注 `ns/op`、`allocs/op`。
- 覆盖率：`go test ./... -coverprofile=cover.out` + `go tool cover -func cover.out`。
- **Playground**：`playground/02_tooling_runner`（命令执行器）、`03_http_middleware`（Logging middleware）、`04_observability`（expvar Registry）。

---

## 09. HTTP & Observability（03_engineering/notes）

- Middleware 链使用 `func(next http.Handler) http.Handler` 模式，顺序常见：RequestID → Logging → Metrics → Recovery → Auth。
- `http.Server` 配置 `ReadTimeout/WriteTimeout/IdleTimeout` 并通过 `srv.Shutdown(ctx)` 优雅退出。
- 观测：`promhttp.Handler()` 暴露 `/metrics`，`import _ "net/http/pprof"` 提供 `/debug/pprof`。

---

## 10. CLI + Service Patterns（04_projects/notes）

- `01_cli_service/internal/bridge` 统一 DTO/Client/Router，CLI 与 API 共享协议，便于未来扩展。
- CLI 配置优先级：默认值 → 配置文件 → env → flag；解析后校验并记录。
- 集成测试：`httptest.NewServer` 运行 API，CLI 注入 fake HTTP client + golden files；Docker Compose 编排 CLI job + API + 数据源。
- 部署 checklist：`make lint/test/build`, `docker build`, `docker compose up`, Release 附运行/监控说明。
- **Playground**：`playground/01_cli_service`（主项目）、`02_ingest_pipeline`（日志解析/聚合）、`03_query_service`（查询 handler）。
