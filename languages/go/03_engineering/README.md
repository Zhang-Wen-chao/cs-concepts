# 阶段 3 · 工程化

> 目标：掌握 Go Modules、测试/基准、HTTP 服务、配置与可观测性、部署流程，交付具备 CRUD + 观测能力的 Todo API（`playground/01_todo_api`）。

## 学习闭环
| 步骤 | 资料 | 产出 |
| --- | --- | --- |
| 阅读 | `notes/01_modules_tooling.md` ~ `05_deployment.md` + 官方 Modules Guide / net/http docs | 工程笔记、命令速记 |
| 实验 | `playground/01_todo_api` | Repository + Handler + Integration Test + Benchmark |
| 运行 | `go test ./... -run Test -bench . -benchmem -cover` | 单测/子测/基准/覆盖率齐全 |
| 记录 | `go_cheatsheet.md` 工程化篇 | 常用命令、部署脚本、调优提示 |

## Checklist
- [ ] 熟悉 `go env GOPATH`, `go env GOMOD`, `go list -m all`，会处理 `replace`/`vendor`。
- [ ] 形成 `fmt/vet/golangci-lint` 一键脚本（Makefile 或 Taskfile），默认执行。
- [ ] 测试心智模型：table-driven + subtest + mock 接口 + `testing.T.Helper`。
- [ ] Benchmark：`-bench`, `-benchmem`, `benchstat` 流程写进 `notes/02_testing_benchmark.md`。
- [ ] HTTP：了解 middleware 链、context 取消、`http.Server` 生命周期，能写健康检查/metrics。
- [ ] 配置与可观测性：flag + env + config file（如 `viper`）组合；日志/metrics/trace 接入路线清晰。
- [ ] Todo API：CRUD + 状态过滤 + 分页；`httptest` 集成测试 + storage 层单测；覆盖率 ≥ 70%。
- [ ] Dockerfile + Makefile：支持 `make test`, `make run`, `docker build`, `docker run -p`.

## Playground 模块
- `01_todo_api` —— RESTful Todo 服务骨架。
- `02_tooling_runner` —— 注入式命令执行器，示范如何封装 fmt/vet/test。
- `03_http_middleware` —— Logging middleware，可直接挂到 `http.ServeMux`。
- `04_observability` —— `expvar` 计数器注册表 + `/metrics` Demo。

## 验收方式
```bash
cd languages/go/03_engineering/playground
go fmt ./...
go vet ./...
golangci-lint run ./...    # 若未安装，可先使用 go vet
go test ./... -run Test -bench . -benchmem -cover
make run OR go run 01_todo_api
```
- `/healthz` 返回 200，`/metrics` 暴露基础指标（可用 `expvar`/`prometheus`）。
- Docker 镜像可通过环境变量配置端口、日志级别。

## 复盘要点
- 模块/依赖策略：是否需要 private module？如何在 monorepo 中隔离。
- 测试金字塔：单测/集测/基准覆盖是否均衡。
- 部署体验：单条命令完成 build/test/package 吗？下一阶段如何自动化。
