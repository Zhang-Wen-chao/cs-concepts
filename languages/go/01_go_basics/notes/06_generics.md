# 06 · 泛型速览

> 资料：Go 官方教程「Generics in Go」、proposal 文档、`constraints` 包。

## 基本语法
```go
func Map[T, R any](in []T, f func(T) R) []R {
    out := make([]R, len(in))
    for i, v := range in {
        out[i] = f(v)
    }
    return out
}
```
- 类型参数写在函数名后 `[T any]`。
- 约束 `any` 等价于空接口；可使用 `constraints.Ordered` 等预定义约束。

## 类型参数推断
- 调用时通常省略类型参数：`Map([]int{1,2}, func(i int) string { ... })`。
- 仅当无法推断或想覆盖默认推断时显式写类型：`Map[string](...)`。

## 实例：可比较集合
```go
type Set[T comparable] map[T]struct{}

func (s Set[T]) Has(v T) bool { _, ok := s[v]; return ok }
```
- `comparable` 约束允许在 map key 或 `==` 中使用。

## 性能与可读性
- 泛型适合容器/算法复用，避免 copy/paste；业务对象通常仍用具体类型。
- 程序生成的代码针对每个使用的类型实例化，不会像 C++ 模板产生爆炸性符号。

## Checklist
- [ ] 能写出一个泛型容器或算法（Map/Filter/Set）。
- [ ] 清楚 `comparable`、`constraints.Ordered`、`~int`（底层类型约束）的用途。
- [ ] 知道何时保持简单（接口/具象类型）而不是过度泛化。
