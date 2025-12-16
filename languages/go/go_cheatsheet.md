# Go 核心小抄

> 阶段 1：Go 基础（进行中）

---

## 01. Go Mindset（notes/01_go_mindset.md + playground/mindset/greet）

**核心理念**
1. 工具先行：写任何 Go 代码前跑 `gofmt -w .`、`go test ./...`，风格与正确性交由工具兜底。
2. 声明即文档：导出函数/类型必须写 doc comment（`// Greeter ...`），否则 `go doc`/pkg.go.dev 无法生成说明。
3. Less is More：组合优于继承，遇到并发问题优先 goroutine + channel（“share memory by communicating”）。

**实践（flag + 表驱动测试）**
- `flag.String(name, default, help)` 返回 `*string`，先 `flag.Parse()` 再 `*name` 解引用。
  ```go
  name := flag.String("name", "gopher", "target")
  flag.Parse()
  fmt.Println(greeting(*name, *lang))
  ```
- `:=` 是短变量声明，只能在函数内部使用；`greeting(*name, *lang)` 返回 `(string, error)`，用 `msg, err := ...` 接收。
- 表驱动测试：`cases := []struct{ ... }{... }` + `t.Run` 循环，测试文件命名 `*_test.go` 并可用 `t.Parallel()` 并行执行。

---

## 02. Syntax Basics（notes/02_syntax_basics.md + playground/syntax_basics/stats）

1. 变量/常量：`var count int` 自动赋零值；函数内可用 `:=`。常量在编译期确定，例如 `const Pi = 3.14`。
2. 多返回值：`func describeNumbers(nums []int) (Report, error)` —— 先判断 `err != nil` 再使用结果；命名返回值仅在极短函数中使用。
3. 控制流：`if/for/switch` 都支持短语句（`if v, err := ...; err != nil { ... }`），`switch` 默认 `break`。切片遍历形态 `for i, v := range nums`，可用 `_` 忽略索引。

---

## TODO
- [ ] 03 · Collections
- [ ] 04 · Struct & Interface
- [ ] 05 · Error Handling
- [ ] 06 · Concurrency Toolkit
- [ ] 07 · Tooling & Testing
