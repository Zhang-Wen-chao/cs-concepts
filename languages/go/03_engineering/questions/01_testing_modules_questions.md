# Go 工程化 — 问题版（自测用）

## Q1

`go test` 的 table-driven test 怎么写？

<details>
<summary>答案</summary>

```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name string
        a, b int
        want int
    }{
        {"positive", 1, 2, 3},
        {"negative", -1, -2, -3},
        {"zero", 0, 0, 0},
    }
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            if got := Add(tt.a, tt.b); got != tt.want {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

这是 Go 的标准测试风格。好处：加一个测试 case 只需要加一行 struct 字面量。

</details>

## Q2

`go mod` 的核心命令有哪些？

<details>
<summary>答案</summary>

```bash
go mod init <module>   # 初始化新模块
go mod tidy            # 清理依赖（添加缺失的，移除未用的）
go mod download        # 下载依赖到本地缓存
go mod vendor          # 把依赖拷贝到 vendor 目录
go list -m all          # 列出所有依赖
```

Go 的依赖管理比 C++（CMake）简单得多，比 Python（pip/poetry）也简单。

</details>

## Q3

Go 的 benchmark 怎么写？`-bench=.` 和 `-benchmem` 是什么意思？

<details>
<summary>答案</summary>

```go
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(1, 2)
    }
}
```

```bash
go test -bench=. -benchmem
```

- `-bench=.` — 运行所有 benchmark（`.` 是 regex，匹配所有）
- `-benchmem` — 显示每次操作的内存分配次数和大小

输出示例：
```
BenchmarkAdd-8    1000000000    0.3 ns/op    0 B/op    0 allocs/op
```

- `-8`：8 个 CPU 核心
- `1000000000`：迭代次数
- `0.3 ns/op`：每次操作耗时
- `0 B/op`：每次操作分配的内存
- `0 allocs/op`：每次操作的内存分配次数

</details>

## Q4

`go vet` 和 `golangci-lint` 有什么区别？

<details>
<summary>答案</summary>

- `go vet` — Go 自带的静态分析工具，检查可疑结构（如 Printf 参数不匹配、无用的赋值）。官方维护，精确但检查项有限
- `golangci-lint` — 第三方工具，聚合了 100+ 个 linter，可配置。更严格，检查范围更广

用法：
```bash
go vet ./...                   # 官方
golangci-lint run ./...        # 第三方
```

</details>

## Q5

Go 的 `http.Handler` 接口怎么自定义中间件？

<details>
<summary>答案</summary>

```go
// Handler 接口定义
type Handler interface {
    ServeHTTP(ResponseWriter, *Request)
}

// 中间件模式：接收一个 Handler，返回一个 Handler
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        log.Printf("%s %s", r.Method, r.URL.Path)
        next.ServeHTTP(w, r)
    })
}

// 使用
mux := http.NewServeMux()
mux.HandleFunc("/api", apiHandler)
handler := LoggingMiddleware(mux)
http.ListenAndServe(":8080", handler)
```

</details>
