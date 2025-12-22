# 04 · 配置与可观测性

> 资料：`spf13/cobra` + `viper` 文档、uber-go/zap、prometheus client_golang、`net/http/pprof`。

## 配置优先级
1. **flag**：`flag.String(\"config\", \"config.yaml\", \"config file\")`
2. **环境变量**：使用 `os.LookupEnv` 或 `viper.AutomaticEnv()`。
3. **配置文件**：YAML/JSON/TOML；启动时读取并覆盖默认值。
- 推荐模式：默认值 → env → flag → config file（或反之，根据团队习惯）。

## 日志
- `log/slog`（Go 1.21+）或 `uber-go/zap`。
- 结构化字段：`logger.Info(\"task finished\", \"task_id\", id, \"duration\", d)`.
- 日志级别通过 flag/env 控制，生产环境输出 JSON，开发环境输出彩色文本。

## 指标
- `promhttp.Handler()` 暴露 `/metrics`。
- 请求计数：`prometheus.NewCounterVec`；延迟：`HistogramVec`。
- 若不引入 prometheus，可先使用 `expvar` 暴露基础数值。

## Profiling & Tracing
- `import _ \"net/http/pprof\"` 并在 `/debug/pprof/` 提供性能分析。
- `go test -run Test -cpuprofile cpu.out` 捕获 CPU Profile。
- OpenTelemetry（`go.opentelemetry.io/otel`）用于分布式追踪；Stage 4 可接入。

## Checklist
- [ ] CLI flag + env + config file 三层合并实现，并写测试覆盖。
- [ ] 日志统一通过接口输出，方便在 CLI/API 之间复用。
- [ ] `/metrics`、`/debug/pprof` 路径在本地和容器环境都可访问。
