# Go 学习路径

> 目标：掌握 Go 语法、并发、工程化与综合交付，快速产出 CLI + Service 组合项目。

## 🎯 学习阶段总览

| 阶段 | 目录 | 说明 | 核心产出 |
| --- | --- | --- | --- |
| 01 · Go 基础 | `01_language_foundations/` | 思维方式 + 语法复习 | `01_mindset`, `02_syntax_basics`, `go_cheatsheet.md` |
| 02 · 并发 | `02_concurrency/` | goroutine/channel/context/sync/errgroup | worker-pool 爬虫 |
| 03 · 工程化 | `03_engineering/` | 模块、测试、HTTP、观测、部署 | Todo API + Makefile/Docker |
| 04 · 综合项目 | `04_projects/` | CLI + Service 组合交付 | 日志分析 CLI + API + 文档 |

每个阶段目录包含：
- `README.md`：任务清单、验收标准。
- `notes/`：阅读笔记，链接官方资料。
- `playground/`：对应代码和测试。

## 📖 学习闭环
```
1. 📄 看文档    进入 stage/notes，5-10 分钟速读 + 标记疑问
   ↓
2. 💻 看代码    对应 stage/playground，写出最小示例
   ↓
3. 🚀 运行       go fmt ./... && go test ./... && go test -race ./... && go run ./cmd/...
   ↓
4. 📝 小抄       go_cheatsheet.md 中补录套路 + 指令
```
- 练习时间控制在 15~20 分钟，超时就拆解子问题。
- `go fmt ./...`, `go test ./...`, `golangci-lint run` 作为默认验收命令。

## 🧭 阶段任务速览

### 阶段 1 · Go 基础（`01_language_foundations`）
- 阅读：`notes/01_mindset.md`, `02_syntax_basics.md`，Go Tour Basics/Flow control/Functions。
- 实践：`playground/01_mindset`（flag + table test）、`playground/02_syntax_basics`（算法 + benchmark 雏形）。
- 验收：`cd 01_language_foundations/playground && go fmt ./... && go test ./...`.
- 复盘：写下工具链（go fmt/test）、语法惯性、`go_cheatsheet.md` 更新点。

### 阶段 2 · 并发（`02_concurrency`）
- 阅读：`notes/01_goroutines.md` ~ `05_errgroup_rate_limiting.md`。
- 实践：`playground/01_crawler`（worker pool + context + 限流 + 重试）。
- 验收：`go test ./... && go test -race ./... && go run 01_crawler --urls 01_crawler/fixtures/urls.txt`.
- 输出：在 `go_cheatsheet.md` 新增 goroutine/channel/context/sync/errgroup 速记。
- 额外演练：`playground/02_context_guard`, `03_sync_limiter`, `04_errgroup_pipeline`。

- ### 阶段 3 · 工程化（`03_engineering`）
- 阅读：`notes/01_modules_tooling.md` ~ `05_deployment.md`。
- 实践：`playground/01_todo_api`（Repository + Handler + Server）。
- 验收：`go vet`, `golangci-lint`, `go test ./... -bench . -benchmem -cover`, `docker build`.
- 输出：Makefile/Taskfile、Dockerfile、README、coverage。
- 额外演练：`playground/02_tooling_runner`, `03_http_middleware`, `04_observability`。

### 阶段 4 · 综合项目（`04_projects`）
- 阅读：`notes/01_project_brief.md` ~ `03_integration_testing.md`，完成需求/架构/测试计划。
- 实践：`playground/01_cli_service`（`--mode=api|cli` 切换）、`02_ingest_pipeline`、`03_query_service`。
- 验收：`go test ./... -race`, `go run 01_cli_service --mode=api`, `go run 01_cli_service --mode=cli --query error`, `docker compose up`.
- 输出：项目 README、架构图、操作指南、复盘记录。

更多细节参见各阶段 README。

## 🔧 快速开始（通用）
```bash
# 1. 安装 Go 1.22+
brew install go   # 或到 https://go.dev/dl/ 下载

# 2. 进入目标阶段
cd languages/go/01_language_foundations/playground

# 3. 运行练习
go fmt ./...
go test ./...
go run 01_mindset --name Gopher --lang en
```
- 其他阶段同理：切换到 `02_concurrency/playground` 等目录再执行命令。
- 若需初始化新模块：`go mod init github.com/aaron/cs-concepts/<module-name>`。

## 📅 复盘与进度追踪

| 阶段 | 完成日期 | 产出路径 | 复盘要点 |
| --- | --- | --- | --- |
| 阶段 1 | yyyy-mm-dd | `01_language_foundations/playground` | 例：切片共享导致 bug，已用 `copy` 修复 |
| 阶段 2 |  |  |  |
| 阶段 3 |  |  |  |
| 阶段 4 |  |  |  |

复盘提示：学到了什么？踩坑/定位方法？下周计划？是否要与 `languages/cpp` 进度对齐？

## 📖 推荐资料
1. 《The Go Programming Language》（Donovan & Kernighan）
2. Go Tour：https://tour.golang.org/
3. Go Blog & Go by Example
4. GoTime Podcast、Ardan Labs Blog（工程实践）
5. TopGoer 中文教程：https://www.topgoer.com/

---
保持与 `languages/cpp` 同步：每完成一个阶段就在复盘表更新日期 + 产出链接。
