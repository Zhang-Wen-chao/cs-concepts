# Go 学习路径

> 目标：掌握 Go 的语法基础、并发原语和工程化实践，能写出可靠的后端 / 工具程序。

## 🎯 学习目标
- 理解 Go 的语言哲学（简单、组合、面向并发）。
- 熟悉核心语法：类型、切片、map、接口、错误处理。
- 掌握 goroutine、channel、context 等并发原语。
- 会用 Go Modules、testing、lint、profiling 等工程化工具。
- 完成至少两个实战项目（CLI 工具 + Web/service）。

## 📚 学习路径

### 阶段 1 · Go 基础（约 1 周）
- [ ] 01_go_mindset.md —— Go 的设计哲学 & 与 C++/Python 的差异
- [ ] 02_syntax_basics.md —— 变量、流程控制、函数、多值返回
- [ ] 03_collections.md —— array/slice/map、range、拷贝 vs 引用
- [ ] 04_struct_interface.md —— 组合、方法集、接口与鸭子类型
- [ ] 05_error_handling.md —— error 接口、`errors.Is/As`、panic/recover

**实践：**
- [ ] 迷你 CLI：实现一个 `greet` 工具，演示 flag 解析、字符串处理。

### 阶段 2 · 并发（约 1 周）
- [ ] 01_goroutine.md —— goroutine 生命周期、调度器
- [ ] 02_channel.md —— 无缓冲 vs 有缓冲、select
- [ ] 03_context.md —— cancel、deadline、value 传递
- [ ] 04_sync_primitives.md —— sync.Mutex/RWMutex/WaitGroup
- [ ] 05_error_group.md —— errgroup、并发任务聚合

**实践：**
- [ ] 并发爬虫：给定 URL 列表，开启 goroutine 抓取并统计响应时间。

### 阶段 3 · 工程化（约 1-2 周）
- [ ] 01_modules_tooling.md —— go mod、go fmt、go vet、lint
- [ ] 02_testing.md —— testing 包、table-driven、benchmark、mock
- [ ] 03_http_basics.md —— net/http、handler、middleware
- [ ] 04_config_observability.md —— flag/env/config、zap/logrus、pprof

**实践：**
- [ ] RESTful API：实现 todo 服务（增删改查 + 内存存储）。

### 阶段 4 · 综合项目（约 1 周）
- [ ] CLI + Service 组合项目：例如“日志分析 + HTTP 查询”工具。
- [ ] 与现有 C++/Python 组件集成（通过 gRPC/HTTP）。

## 🚀 快速开始
```bash
# 1. 安装 Go 1.22+
https://go.dev/dl/

# 2. 创建 playground 目录
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
```

## 📖 推荐资料
1. 《The Go Programming Language》（A. Donovan）
2. Go 官方 Tour：https://tour.golang.org/
3. Go Blog & Go by Example：实践导向示例

---
记录每个阶段的完成日期，和 `cpp` 路径一样保持同步，方便回顾学习节奏。
