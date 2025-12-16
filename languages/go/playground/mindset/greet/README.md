# greet CLI

演示如何用 Go flag 包和 table-driven test 构建最小 CLI。

> table-driven test：先把多组输入/期望写进一个表（`[]struct{...}`），再用同一段测试逻辑循环跑完，便于按需增删场景。

## 用法

```bash
go run . --name=Go --lang=en
```

可选语言：`zh`（默认）、`en`。未提供 `--name` 时默认 `gopher`。

`--name`、`--lang` 与 Python CLI 的 `--name`、`--lang` 含义相同，都是 long option，`flag` 包负责解析并填入变量。

## 开发流程

```bash
gofmt -w .
go test ./...
go run .
```
