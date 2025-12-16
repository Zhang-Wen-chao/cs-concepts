# Go 学习路径

> 目标：掌握 Go 的语法基础、并发原语和工程化实践，能写出可靠的后端 / 工具程序。

## 🎯 学习目标
- 理解 Go 的语言哲学（简单、组合、面向并发）。
- 熟悉核心语法：类型、切片、map、接口、错误处理、泛型。
- 掌握 goroutine、channel、context、sync、errgroup 等并发原语。
- 会用 Go Modules、testing、lint、profiling、observability 等工程化工具。
- 完成至少两个实战项目（CLI 工具 + Web/service），并能与其他语言组件集成。

## 📖 学习流程

Go 也沿用 C++ 学习路径里的“文档 → 示例 → 运行 → 小抄”闭环，确保每个主题都学完可落地。

```
1. 📄 看文档      notes/01_go_mindset.md（Effective Go + Less is More 摘要）
   5-10 分钟通读要点，标记疑惑，必要时回到原文
   ↓
2. 💻 看代码      playground/mindset/main.go
   查看/编写示例，理解如何将理念落地
   ↓
3. 🚀 运行代码    gofmt -w . && go test ./... && go run .
   每次练习都格式化 + 测试 + 运行，确保输出与预期一致
   ↓
4. 📝 记录小抄    go_cheatsheet.md
   一句话总结 + 最小代码片段，阶段复习直接查阅
```

**关键：**
- “看文档 / 看代码 / 运行 / 小抄”一一对应，完成一套流程再进入下一个主题。
- 每个主题目标 15-20 分钟，超过 30 分钟就拆分成更小的子问题。
- `gofmt`、`go test ./...`、`golangci-lint run` 作为默认验收流程，保持和工程实践一致。

## 📚 学习路径

每个阶段都用“阅读 / 练习 / 验收”结构，确保勾选即代表真正学完。

### 阶段 1 · Go 基础（约 1 周）
- [ ] Go mindset —— 阅读 [Effective Go（前 3 章）](https://go.dev/doc/effective_go) + Go Blog「Less is More」，理解少抽象、组合优于继承。
- [ ] 语法基础 —— 在 [Go Tour](https://go.dev/tour/welcome/1) 完成 Basics、Flow control、Functions，并动手解小练习；补充笔记见 `notes/02_syntax_basics.md`。
- [ ] 集合与引用语义 —— 研读 Go Blog「[Go Slices: usage and internals](https://go.dev/blog/slices-intro)」，对比 array/slice/map 的拷贝与共享。
- [ ] 组合与接口 —— 阅读 [Methods and interfaces](https://go.dev/tour/methods/1) + Go by Example: Structs/Interfaces，理解方法集、鸭子类型。
- [ ] 错误处理 —— 查阅 `errors` 包文档以及「[Working with Errors in Go](https://go.dev/blog/go1.13-errors)」，练习 `errors.Is/As`、`fmt.Errorf("%w")`、`panic/recover`。
- [ ] 类型与泛型补充 —— 浏览「[Generics in Go](https://go.dev/doc/tutorial/generics)」，了解类型参数、零值、逃逸分析的直观示例。

**实践：迷你 CLI `greet`**
- [ ] 使用 `flag` 或 `cobra` 解析 `--name`、`--lang`，默认输出中文/英文问候。
- [ ] 输出格式通过 table-driven test 覆盖（`go test ./...` 必须通过）。
- [ ] README 记录用法与示例输出，`golangci-lint run`（或 `go vet`）无告警。
- [ ] 目录建议：`playground/mindset/greet`；完成后在 `go_cheatsheet.md` 添加“CLI flag + table test”条目。

### 阶段 2 · 并发（约 1 周）
- [ ] goroutine 调度 —— 阅读官方文档「[Goroutines](https://go.dev/tour/concurrency/1)」+ Go Blog Scheduling 图解。
- [ ] channel 模式 —— 完成 Tour 中 Channel 章节 + Go by Example: Timers/Tickers/Worker Pools，重点练 select、缓冲 vs 非缓冲。
- [ ] context —— 研读 [`context` 包 blog](https://go.dev/blog/context) + `context.WithCancel/Timeout` 用法，了解 value 传递边界。
- [ ] 同步原语 —— 实验 `sync.Mutex/RWMutex/WaitGroup/Once`、`atomic.Value`，理解适用场景。
- [ ] errgroup & 限流 —— 阅读 `golang.org/x/sync/errgroup` 文档，顺便了解 `semaphore`/`rate` 包以控制并发度。

**实践：并发爬虫**
- [ ] 输入 URL 列表，使用 worker pool + context 超时控制，总并发 <= 20。
- [ ] 统计响应时间/状态码，输出 JSON 或表格；失败重试 1 次并记录错误。
- [ ] 完整测试：为抓取逻辑提供 fake server 或 `httptest.Server`；`go test -race ./...` 必须通过。

### 阶段 3 · 工程化（约 1-2 周）
- [ ] Modules & Tooling —— 深入 [Go Modules Guide](https://go.dev/doc/modules/managing-dependencies)，练习 `replace`、`vendor`，使用 `go fmt`, `go vet`, `golangci-lint`, `air`.
- [ ] Testing & Benchmark —— 学习 [Testing pkg](https://pkg.go.dev/testing)、table-driven、subtests、mock 接口，掌握 `-run/-bench/-benchmem`、`benchstat`、`coverage`。
- [ ] HTTP & Middleware —— 阅读 [net/http](https://pkg.go.dev/net/http) 官方示例，了解 `http.Server` 生命周期、context、`chi`/`gin` 等 router。
- [ ] 配置 / 观测性 —— 实践 `flag` + env + config file（如 `viper`），接入 `zap/logrus` 日志，使用 `pprof`, `expvar`, `prometheus` 指标；尝试 `trace` 或 `pprof` 分析。
- [ ] 部署准备 —— 编写 `Makefile`/`Taskfile`、Dockerfile，熟悉 `ENV`/`ARG`、多阶段构建。

**实践：RESTful Todo API**
- [ ] CRUD + 过滤查询（分页/状态）在内存或 SQLite 中实现。
- [ ] 编写 integration test（可用 `httptest`）与 benchmark，覆盖率 >= 70%。
- [ ] 暴露 `/healthz`、`/metrics`，提供 swagger 或 simple markdown 文档。
- [ ] 容器化运行：`docker build` + `docker run` 成功，支持配置化端口/日志级别。

### 阶段 4 · 综合项目（约 1 周）
- [ ] 设计 “CLI + Service” 组合，如「日志分析 CLI + HTTP 查询服务」，定义需求、技术栈、交付物。
- [ ] 规划里程碑：数据采集层 → 分析/持久化 → API/CLI 输出 → 集成测试。
- [ ] 与现有 C++/Python 组件对接：优先 gRPC/HTTP，明确 proto/JSON schema，验证互操作性与性能。
- [ ] 最终交付包含：README、架构图、部署脚本、基准数据、回顾笔记（问题 & 改进）。

## 🔧 快速开始
```bash
# 1. 安装 Go 1.22+
https://go.dev/dl/

# 2. 创建 playground 目录（用于阶段练习）
cd languages/go && mkdir -p playground && cd playground

# 3. 初始化模块
go mod init github.com/yourname/go-playground

# 4. Hello World
cat <<'HELLO' > main.go
package main
import "fmt"
func main() {
    fmt.Println("Hello, Go!")
}
HELLO

go run .
go test ./...   # 默认在每个练习目录都跑一次
```

## 📅 复盘与进度记录
在 README 底部追加最新记录即可，保持与 `languages/cpp` 相同节奏。

| 阶段 | 完成日期 | 产出目录/链接 | 复盘要点 |
| --- | --- | --- | --- |
| 阶段 1 | yyyy-mm-dd | languages/go/playground/cli | 例：切片共享导致 bug，已用 copy 修复 |
| 阶段 2 |  |  |  |
| 阶段 3 |  |  |  |
| 阶段 4 |  |  |  |

复盘要回答：本周学到了什么？踩坑/定位方法？下一步计划？是否需要与 C++ 进度对齐？

## 📖 推荐资料
1. 《The Go Programming Language》（A. Donovan）
2. Go 官方 Tour：https://tour.golang.org/
3. Go Blog & Go by Example：实践导向示例
4. GoTime Podcast + Ardan Labs Blog（工程实践/调优案例）

---
记录每个阶段的完成日期，和 `cpp` 路径一样保持同步，方便回顾学习节奏。
