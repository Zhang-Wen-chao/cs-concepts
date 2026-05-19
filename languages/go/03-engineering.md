# Go 工程实践

## 模块与工具链
```bash
go mod init example.com/foo     # 初始化模块
go mod tidy                      # 清理依赖
go mod vendor                    # 离线依赖
go list -m all                   # 查看依赖图
go env -w GOPRIVATE=github.com/yourorg/*   # 私有仓库
replace old => ../local          # 替换为本地包调试
```
- v2+ 必须更新 module path：`module example.com/foo/v2`

## 测试

| 类型 | 命令 | 目标 |
|------|------|------|
| 单元测试 | `go test ./...` | 表驱动 + subtest + `t.Parallel()` |
| Benchmark | `go test -bench Foo -benchmem` | `ns/op`, `B/op`, `allocs/op` |
| Coverage | `go test -coverprofile=coverage.out` | 函数级覆盖率 |
| Race | `go test -race` | 检测数据竞争 |

- Mock：注入接口 + `httptest.Server` / `t.Cleanup`
- 对比 benchmark：`benchstat old.txt new.txt`
- 推荐工具：`cmp.Diff`（`google/go-cmp`）比较结构体

## HTTP 服务
```go
srv := &http.Server{
    Addr:         ":8080",
    Handler:      mux,
    ReadTimeout:  5 * time.Second,
    WriteTimeout: 10 * time.Second,
}
// 优雅关闭：signal → srv.Shutdown(ctx)
```

### Middleware 链
顺序：RequestID → Logging → Metrics → Recovery → Auth

```go
type Middleware func(http.Handler) http.Handler
// 每个 middleware 包装 next.ServeHTTP
```

- 标准库 `http.ServeMux` 够用；复杂路由可用 `go-chi/chi/v5`

## 配置与可观测性

| 维度 | 工具 | 端点 |
|------|------|------|
| 配置 | flag + env + YAML（viper） | — |
| 日志 | `log/slog`（1.21+）或 `uber-go/zap` | 结构化字段 |
| 指标 | `prometheus/client_golang` | `/metrics` |
| Profiling | `net/http/pprof` | `/debug/pprof/` |
| 追踪 | OpenTelemetry | 分布式 tracing |

- 配置优先级：默认值 → env → flag → 配置文件（按团队习惯）

## 构建与部署

```dockerfile
# 多阶段构建
FROM golang:1.22 AS build
RUN go build -o /out/server .

FROM gcr.io/distroless/base
COPY --from=build /out/server /server
ENTRYPOINT ["/server"]
```

- 最终镜像 < 50MB（distroless/alpine）
- 健康检查：k8s `readinessProbe` / `livenessProbe`
- Build info：`-ldflags "-X main.version=$(git rev-parse --short HEAD)"`

## playground
- [`03_engineering/playground`](./03_engineering/playground/) — todo API、tooling、middleware、observability
- [`04_projects/playground`](./04_projects/playground/) — CLI service、ingest pipeline、query service
