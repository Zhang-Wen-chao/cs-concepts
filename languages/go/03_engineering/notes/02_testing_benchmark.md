# 02 · Testing & Benchmark

> 资料：`testing` 包、Go Blog「Table-driven tests」「Subtests」、Dave Cheney「Practical Go Benchmarks」。

## 单元测试套路
```go
func TestFoo(t *testing.T) {
    t.Parallel()
    cases := []struct {
        name string
        in   input
        want output
        err  error
    }{
        {name: \"ok\", ...},
    }
    for _, tc := range cases {
        tc := tc
        t.Run(tc.name, func(t *testing.T) {
            t.Parallel()
            got, err := Foo(tc.in)
            if !errors.Is(err, tc.err) { ... }
            if diff := cmp.Diff(tc.want, got); diff != \"\" { ... }
        })
    }
}
```
- `t.Helper()` 封装重复断言。
- `cmp.Diff`（`google/go-cmp`）帮助比较结构体。

## Mock 策略
- 倾向注入接口（如 `Repository`、`Clock`、`HTTPClient`）。
- 使用 `t.Cleanup` 释放临时资源（文件/服务器）。
- 对第三方服务采用 `httptest.Server` 或 fake 实现。

## Benchmark 基础
```go
func BenchmarkFoo(b *testing.B) {
    for i := 0; i < b.N; i++ {
        _ = Foo(payload)
    }
}
```
- 运行：`go test -bench Foo -benchmem -run ^$`.
- 基准需剥离 I/O，关注 `ns/op`, `B/op`, `allocs/op`。
- 对比版本可用 `benchstat old.txt new.txt`。

## Coverage & Profiles
- `go test ./... -coverprofile=coverage.out`.
- `go tool cover -func=coverage.out` 查看函数覆盖率。
- `go test -run TestFoo -cpuprofile cpu.out` + `go tool pprof cpu.out`.

## Checklist
- [ ] 表驱动测试 + subtest + `t.Parallel()` 在大多数测试中使用。
- [ ] Benchmark 至少覆盖一个热点函数，并记录 baseline。
- [ ] 通过覆盖率报告定位未测试路径，补齐关键分支。
