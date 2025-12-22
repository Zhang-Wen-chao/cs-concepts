# 03 · 集成测试与交付

## 测试金字塔
- **单测**：CLI 解析器、Aggregator、API handler 均有 table-driven 测试。
- **集成测试**：使用 `httptest.Server` 启动 API，CLI 使用 fake HTTP client，验证端到端协议。
- **E2E**：docker-compose + `make e2e`，在容器内运行 CLI → API → 查询。

## CLI 测试技巧
- 将 IO 抽象到接口（`type Reader interface { Next() ([]byte, error) }`），测试中注入假数据。
- 对输出使用 golden file：`testdata/cli_report.golden`，用 `cmp.Diff` 比较。

## API 测试技巧
```go
srv := httptest.NewServer(router)
defer srv.Close()

resp, err := http.Post(srv.URL+\"/v1/ingest\", \"application/json\", bytes.NewReader(body))
```
- 使用 `require`/`cmp` 检查状态码与响应体。
- 记录 `rr.Header().Get(\"X-Request-ID\")`，确保中间件运作。

## CI/CD
- GitHub Actions：lint -> test -> docker build -> upload artifact。
- 版本发布：Git tag 触发 `goreleaser`，生成二进制 + Docker 镜像。
- 文档与包：生成 `swagger.json` 或 Markdown，随 release 附件发布。

## Checklist
- [ ] `01_cli_service/cmd/cli` 与 `01_cli_service/cmd/api` 皆有 `testdata/`，并可通过 `go test ./...` 运行。
- [ ] `make e2e` 启动 docker-compose，跑完自动清理容器。
- [ ] Release 说明包含版本、变更、校验和。
