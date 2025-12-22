# 05 · 错误处理

> 资料：Go Blog「Working with Errors in Go」、`errors` 包文档、Go1.20+ `errors.Join`。

## 基本模式
```go
if err := doThing(); err != nil {
    return fmt.Errorf("doThing: %w", err)
}
```
- 总是把上下文信息包裹在错误中，便于排查。
- 仅在无法继续时使用 `panic`；库代码多返回错误，不要 `log.Fatal`。

## 哨兵错误 & `errors.Is`
```go
var ErrNotFound = errors.New("not found")

if errors.Is(err, ErrNotFound) {
    // ...
}
```
- 用 `errors.Is/As` 替代手写类型断言，尤其在链式包裹后。

## 多错误
- `errors.Join(errs...)` 聚合多个错误，配合 `errors.Is` 仍可匹配具体哨兵。
- 第三方库如 `multierror` 可统一展示多行。

## `defer` + 清理
- 在 `return` 之前执行：常用于关闭文件、释放锁。
- 注意 `defer` 捕获的是变量，而非表达式结果；必要时在 `defer` 中使用局部变量。

## Checklist
- [ ] 会定义/使用哨兵错误，并通过 `errors.Is/As` 判断。
- [ ] 了解 `fmt.Errorf("...%w", err)` 与 `%v` 的差别。
- [ ] 明确 panic 与 error 的分界：只有不可恢复/编程错误才 panic。
