# 阶段 1 · Go 基础

> 目标：完成 Go 思维方式 + 核心语法（语法、集合、组合/接口、错误、泛型）复习，并交付 6 组可运行示例：`01_mindset/greet`、`02_syntax_basics/stats`、`03_collections/slices`、`04_interfaces/shapes`、`05_errors/validator`、`06_generics/transform`。

## 学习闭环
1. **看文档**：`notes/01_go_mindset.md` ~ `notes/06_generics.md`（必要时回溯 Go Tour/Go Blog）。
2. **看代码**：阅读/扩展 `playground/*` 示例，保证每份笔记对应可运行代码 + 测试。
3. **运行校验**：`go fmt ./... && go test ./... && go run <demo>`（存在 `cmd/demo` 时顺手运行）。
4. **记录小抄**：把关键套路汇总到 `go_cheatsheet.md`，复盘时直接查阅。

## Checklist
- [x] Mindset：`notes/01_go_mindset.md` + `playground/01_mindset/greet`（flag + 表驱动测试）。
- [ ] Syntax：`notes/02_syntax_basics.md` + `playground/02_syntax_basics/stats`（if/for/switch + 多返回值）。
- [ ] Collections：`notes/03_collections.md` + `playground/03_collections/slices`（切片分块 + map 合并 + 拷贝策略）。
- [ ] Structs & Interfaces：`notes/04_structs_interfaces.md` + `playground/04_interfaces/shapes`（方法集、type switch、鸭子类型）。
- [ ] Error handling：`notes/05_error_handling.md` + `playground/05_errors/validator`（`errors.Is/Join`、哨兵错误）。
- [ ] Generics：`notes/06_generics.md` + `playground/06_generics/transform`（Map/Filter/Keys 泛型函数）。
- [ ] `go_cheatsheet.md` 增补以上主题的速记条目，并附代码路径。

## 验收方式
```bash
cd languages/go/01_go_basics/playground
go fmt ./...
go test ./...
go run 01_mindset/greet --name "Gopher" --lang zh
go run 03_collections/slices/cmd/demo
go run 04_interfaces/shapes/cmd/demo
go run 05_errors/validator/cmd/demo
go run 06_generics/transform/cmd/demo
```
- CLI 与 demo 均可运行且输出符合预期。
- 所有测试（含 table-driven）必须通过；如新增 benchmark，用 `go test -bench=. -run=^$` 验证能跑通。

## 复盘要点
- 工具链：提交前是否自动执行 `go fmt/go test`？是否需要 `make`/`just` 快捷方式。
- 语法惯性：哪些地方仍带有其他语言的写法？下一阶段如何避免。
- 小抄：能否仅凭 `go_cheatsheet.md` 复述“切片分块/接口/错误/泛型”示例。
