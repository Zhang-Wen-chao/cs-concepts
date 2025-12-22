# 01 · Modules & Tooling

> 资料：Go Modules Guide, `go help modules`, Russ Cox: “Modules, Packages, and Versions”, golangci-lint docs.

## Modules 命令速查
| 命令 | 说明 |
| --- | --- |
| `go env GOPATH` / `GOMOD` | 确认当前模块根目录 |
| `go list -m all` | 查看依赖图 |
| `go get example.com/pkg@latest` | 升级/添加依赖 |
| `go mod tidy` | 增删 `go.sum` 条目 |
| `go mod vendor` | 在 `vendor/` 存储依赖，配合 CI/offline |
| `replace old => ../local` | 临时替换模块，调试本地包 |

## 版本策略
- 遵循语义化版本（`vMAJOR.MINOR.PATCH`），`v2+` 必须更新 module path（`module example.com/foo/v2`）。
- 私有仓库：使用 GOPRIVATE 环境变量（`go env -w GOPRIVATE=github.com/yourorg/*`）。
- 多模块 monorepo：每个子目录一个 `go.mod`，通过 `replace` 指向 `../` 路径。

## 工具链默认组合
```bash
go fmt ./...
goimports -w .
go vet ./...
golangci-lint run ./...
staticcheck ./...
```
- 建议写入 `Makefile` 或 `Taskfile.yml`：`make lint`, `make test`, `make run`。
- 如果未安装 golangci-lint，至少执行 `go vet` 与 `staticcheck`。

## IDE & 命令行增强
- `dlv debug`（Delve 调试器）调试复杂逻辑。
- `gopls` 提示 + gofumpt = 更严格格式。
- `richgo test` / `gotestsum` 改善测试输出，可在大型项目中快速识别失败。

## Checklist
- [ ] `03_engineering/playground` 内 `go env GOMOD` 指向正确路径。
- [ ] `Makefile`/`Taskfile` 里包含 fmt/vet/lint/test/bench/cover。
- [ ] 知道如何用 `replace` 依赖本地未发布模块，以及何时需要 `vendor`。
