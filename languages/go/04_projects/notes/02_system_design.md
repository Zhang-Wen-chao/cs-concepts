# 02 · 系统设计与数据流

## 模块分层
1. **CLI** (`01_cli_service/cmd/cli`)
   - Input Adapter：`stdin` / 文件 / `tail -f`，支持 glob。
   - Parser：正则或快速切割函数，将日志解析为结构化事件。
   - Aggregator：窗口统计（每 5s 滚动），支持多维指标。
   - Transport：调用 `01_cli_service/internal/bridge.Client` 推送数据。
2. **Service** (`01_cli_service/cmd/api`)
   - HTTP 层：路由、auth、中间件（日志/metrics/限流）。
   - Domain：聚合查询、过滤、排序。
   - Storage：内存（map + RWMutex）或 SQLite（`modernc.org/sqlite`）。
   - Observability：`/healthz`, `/metrics`, `/debug/pprof`.

## 数据流示意
```
tail -F access.log --> CLI Parser --> Aggregator (map[endpoint]metrics)
      |                               |
      +----> local stdout report      +--> HTTP POST /v1/ingest (JSON batch)

API /v1/query?endpoint=/login --> storage snapshot --> JSON response
```

## 并发模型
- CLI：使用 Stage 2 的 worker pool 读取文件块，`errgroup + semaphore` 控制 I/O 并发。
- API：`01_cli_service/internal/bridge` 接受批量数据，使用 `sync.RWMutex` 保护内存指标；背景 goroutine 定期 flush/持久化。

## 接口草案
- `POST /v1/ingest`：`[{ \"endpoint\": \"/login\", \"count\": 42, \"p99\": 120}]`
- `GET /v1/query?endpoint=/login&window=1m`
- `GET /healthz` / `GET /metrics`

## 风险与对策
- 大文件读取：使用 `bufio.Reader` + streaming，避免一次性读取。
- 数据倾斜：对 key 做 sharding（`map[string]*bucket` + 分段锁）。
- 崩溃恢复：CLI 支持 checkpoint（记录最后读取 offset）；API 存储定期落盘。
