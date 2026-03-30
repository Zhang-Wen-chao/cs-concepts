# 03 · HTTP & Middleware

> 资料：`net/http` 文档、Go Blog「Http Server」，Chi/Gin 官方示例，Segment/Cloudflare middleware 模板。

## `http.Server` 生命周期
- 构建：`&http.Server{Addr: \":8080\", Handler: mux, ReadTimeout: 5 * time.Second}`
- 启动：`log.Fatal(srv.ListenAndServe())` 或 `srv.Serve(listener)`
- 优雅关闭：捕获信号后 `srv.Shutdown(ctx)`，释放连接。

## Handler 与 Router
- 标准库 `http.ServeMux` 够用；需高级路由时使用 `github.com/go-chi/chi/v5`。
- handler 签名：`func(w http.ResponseWriter, r *http.Request)`；读取 query/path/body 并返回 JSON。
- 请求 ID：可用 `middleware.RequestID()` 或 `uuid`，写入 context。

## Middleware 模型
```go
type Middleware func(http.Handler) http.Handler

func Logging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        log.Printf(\"%s %s %v\", r.Method, r.URL, time.Since(start))
    })
}
```
- 中间件顺序重要：RequestID → Logging → Metrics → Recovery → Auth。
- 可用装饰器把 `http.ResponseWriter` 包装成记录 status/bytes。

## JSON/错误处理
- 设置 `w.Header().Set(\"Content-Type\", \"application/json\")`。
- 自定义错误结构 `{ \"error\": \"message\", \"code\": \"VALIDATION\" }`。
- `encoding/json` 默认使用结构体标签（`json:\"title\"`）。

## Checklist
- [ ] 了解 `http.Server` 的超时参数（Read/Write/Idle Timeout）以及默认值。
- [ ] 写出 middleware 链并解释执行顺序。
- [ ] 集成 `httptest.NewRecorder()` + `httptest.NewRequest()` 编写 handler 测试。
