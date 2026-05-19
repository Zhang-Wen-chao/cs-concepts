# Go 语言基础

## 设计哲学
- **少即是多**：无继承、无异常、无泛型（1.18前）、无宏。用组合代替继承，用接口代替虚表。
- **工具强制一致性**：`go fmt` / `go vet` / `staticcheck` 是默认流程，个人专注在抽象和工程。
- **"Share memory by communicating"**：goroutine + channel 让同步语义显式化。

## 变量与类型
- `var` 自动零值（int→0, string→"", 指针/interface→nil）；`:=` 短声明只在函数内使用。
- `const` 为编译期常量；未指定类型时按上下文推断。
- 指针存在但没有算术运算；函数默认按值传递（slice/map/channel 共享底层数据）。

## 流程控制
- 只有 `for`：`for init; cond; post {}` / `for cond {}` / `for {}`
- `if`/`switch`/`for` 都支持短语句初始化：`if v := expr; v > 10 { ... }`
- `switch` 默认自动 break，除非 `fallthrough`
- `defer` 在函数返回前 LIFO 执行，适合资源释放

## 集合类型

| 类型 | 语义 | 注意 |
|------|------|------|
| `[N]T` 数组 | 值类型，赋值/传参复制整块 | 不常用 |
| `[]T` 切片 | 引用语义（ptr+len+cap） | `append` 超 cap 时分配新底层数组 |
| `map[K]V` | 引用语义 | 零值 nil 只能读不能写；遍历顺序不稳定 |
| `struct` | 值类型 | 字段大写导出，匿名字段实现组合 |

## 接口（鸭子类型）
- 类型隐式实现接口（无需 `implements` 声明）
- 接口小而专（1-3 方法），`io.Reader` / `http.Handler` 为代表
- 接口由消费方定义，避免在生产者端过度抽象

```go
type Shape interface { Area() float64 }
// Circle/Rectangle 隐式实现 Shape，无需声明
```

## 错误处理
- 多返回值是规范模式：`result, err := doThing()`
- 哨兵错误：`errors.Is(err, ErrNotFound)` — 链式包裹后仍可匹配
- `fmt.Errorf("...%w", err)` 保留错误链；`%v` 只格式化
- `errors.Join` 聚合多错误
- `panic` 仅用于不可恢复的编程错误；库代码不 panic

## 泛型（1.18+）
```go
func Map[T, R any](in []T, f func(T) R) []R
```
- `comparable` 约束用于 map key / `==`
- `constraints.Ordered` 用于 `<`/`>` 
- `~int` 约束底层类型（如 `type MyInt int`）
- 适用于容器/算法复用；业务代码仍优先具体类型或小接口

## playground
- [`01_go_basics/playground`](./01_go_basics/playground/) — 语法基础、集合、接口、错误、泛型示例
