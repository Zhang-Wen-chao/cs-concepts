# syntax_basics/stats

演示变量声明、`if` 短语句、`for range`、多返回值以及 `switch`。

## 亮点
- `describeNumbers` 返回 `(Report, error)`，展示多返回值 + 命名结构。
- `if report, err := describeNumbers(...); err != nil { ... }` 演示短语句。
- `classify` 用不带表达式的 `switch` 处理区间，省去冗余 `break`。

## 开发流程

```bash
go fmt ./...
go test ./...
go run .
```
