# 01 · Go Mindset

> 资料：Effective Go（Introduction / Formatting / Commentary），Go Blog《Less is More》，Rob Pike 演讲。

## 为什么要读
- 避免把 C++/Java 思维直接移植到 Go，提前建立“组合优先、约定优于配置”的语言感。
- 明白工具链（gofmt、godoc）负责风格一致性，个人精力专注在抽象与工程实践。
- 认识 Go 团队“少即是多”的设计哲学，在选型/重构时不再盲目追求语法炫技。

## Effective Go 要点

### Introduction
- Go 借鉴多语言但不追求兼容；要写好 Go，必须接受其语义和约定，避免逐字翻译他语言。
- 代码面向团队协作，命名 / 格式 / 构建流程都要遵循社区标准，减少理解成本。

### Formatting
- “格式问题交给机器”是 Go 的基本规则，`gofmt`/`go fmt ./...` 是唯一正确答案。
- 依赖 `gofmt` 自动对齐注释、格式化 import；若格式结果不满意，重构代码或提 issue，而不是手动对齐。
- 建议开启 IDE “format on save”，并把 `go fmt ./...` 放进提交前的常规流程。

### Commentary
- 默认为 `//` 行注释；`/* */` 仅用于包注释或临时屏蔽大段代码。
- 紧贴声明、无空行的注释会作为 doc comment，被 `go doc` / `pkg.go.dev` 自动抓取。
- Doc comment 应写成完整句子，描述“做什么/返回什么/注意事项”，方便 API 使用者直接阅读。

## Less is More 要点
- Go 有意识地删除复杂特性（继承层级、宏、异常），换取更快编译、低心智负担和一致风格。
- 组合优于继承：通过接口和结构体嵌入拼装行为，使依赖关系简单、易于并发扩展。
- “Share memory by communicating”：用 goroutine + channel 让同步语义显式，减少隐式共享带来的数据竞争。
- 新特性引入极其克制，只在能显著提升工程生产力时才引入；其余交给标准库、工具或框架解决。

## 行动清单
- 写任何 Go 代码前先跑 `go fmt ./...`、`go test ./...`，形成肌肉记忆。
- 准备 `go env GOPATH`、`go version` 输出，确保环境≥1.22。
- 建立 `playground/01_mindset` 与 `go_cheatsheet.md`：代码示例和速记一一对应。
- 给每个导出符号写 doc comment（哪怕是 demo），练习“声明即文档”的思维。
- 复盘：记录“组合 vs 继承”“工具强制一致性”“并发设计”三点对当前工作的影响。
