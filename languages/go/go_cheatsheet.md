# Go 核心小抄

> 阶段 1：Go 基础（进行中）

---

## 01. Go Mindset

**工具优先**
- `gofmt -w .` 统一缩进、注释对齐；发现排版问题第一时间跑它。
- `go test ./...`、`golangci-lint run` 视为默认守门人，保持提交干净。

**声明即文档**
- 所有导出符号写 doc comment：`// Greeter prints ...`，便于 `go doc` 与 pkg.go.dev 生成说明。

**并发哲学**
- “Share memory by communicating” → 优先 goroutine + channel，最后才考虑共享锁。

---

## 02. CLI + Tests（mindset/greet）

**flag 解析**
```go
var name = flag.String("name", "gopher", "target")
flag.Parse()
fmt.Println(greeting(*name, *lang))
```
- `go run . --name=Go --lang=en` 与 Python 的 `--name/--lang` 相同，全部在 `main` 中解析。

**表驱动测试**
- 测试文件必须命名为 `*_test.go`，示例：
```go
func TestGreeting(t *testing.T) {
    cases := []struct{ name, lang, want string }{
        {"Go", "en", "Hello, Go!"},
        {"", "zh", "你好，gopher！"},
    }
    for _, c := range cases {
        got, err := greeting(c.name, c.lang)
        if got != c.want || (err != nil) != false {
            t.Fatalf("greeting(%q,%q)=%q err=%v", c.name, c.lang, got, err)
        }
    }
}
```
- 把输入/期望写成表格（`[]struct{}`），循环 `t.Run` 即可覆盖多场景。

---

## 03. Syntax Basics（syntax_basics/stats）

**变量与零值**
- `var count int` 自动赋 0；函数外只能用 `var`，函数内可用 `:=`。
- 常量是编译期值：`const Pi = 3.14`。

**多返回值 + 错误优先**
```go
func describeNumbers(nums []int) (Report, error) {
    if len(nums) == 0 {
        return Report{}, fmt.Errorf("empty input")
    }
    // ...
    return report, nil
}
```
- 使用结果前先判断 `err`，养成“error-first”习惯。

**控制流**
- 唯一循环：`for`。`if`、`switch` 支持短语句：
```go
if r, err := describeNumbers(nums); err != nil {
    return err
} else {
    fmt.Println(r.Sum)
}

switch {
case sum < 0:
    return "negative"
case sum == 0:
    return "zero"
default:
    return "positive"
}
```
- `switch` 默认 break，不必写 `break`，除非使用 `fallthrough`。

---

## TODO
- [ ] Collections
- [ ] Struct / Interface
- [ ] Error Handling
- [ ] Concurrency Toolkit
- [ ] Tooling & Testing
