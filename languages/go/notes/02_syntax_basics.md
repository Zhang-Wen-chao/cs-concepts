# 02 · Syntax Basics

> 资料：Tour of Go（Basics / Flow control / Functions / Methods），Effective Go 相关章节

## 变量与零值
- `var` 声明会自动赋零值（int→0、string→""、指针/interface→nil），不必手动初始化。
- 使用 `:=` 的短变量声明只能在函数内使用，且左侧必须有新变量。
- `const` 只能保存编译期常量；未指定类型的常量会在赋值时推断。

```go
var count int            // 0
name := "gopher"        // 推断 string
const Pi = 3.14159
```

## 函数与多返回值
- 函数可以返回多个值，常用于“结果 + 错误”。
- 命名返回值可以在函数尾部直接 `return` 省略变量名，但只在逻辑简单时使用。
- 支持可变参数（`func sum(nums ...int)`），调用时 `sum(slice...)` 可展开切片。

```go
func average(nums []int) (float64, error) {
    if len(nums) == 0 {
        return 0, errors.New("empty input")
    }
    total := 0
    for _, n := range nums {
        total += n
    }
    return float64(total) / float64(len(nums)), nil
}
```

## 流程控制
- Go 只有一种循环：`for`。支持 `for init; condition; post {}`、`for condition {}`、`for {}`（无限循环）。
- `if`、`switch`、`for` 语句都支持“短语句”作为初始化部分（如 `if v := expr; v > 10 { ... }`）。
- `switch` 默认会在匹配后自动 `break`，除非显式 `fallthrough`；条件可以留空、也可以是表达式。
- `defer` 在函数返回前执行，按栈顺序 LIFO，非常适合资源释放或日志。

```go
for i := 0; i < 3; i++ { ... }

if result, err := doWork(); err != nil {
    return err
} else {
    fmt.Println(result)
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

## 指针与复合类型（快速预览）
- Go 有指针但没有指针运算，函数默认按值传递；对切片、map、channel 的复制依然共享底层数据。
- `struct` 可嵌套、匿名字段可模拟“组合”。
- 这些内容会在后续章节展开，这里先知道语法形态即可。

## 行动清单
- 用短语句写 `if err := ...; err != nil { ... }`，替换掉外层变量声明。
- 所有新 demo 必须覆盖：`for range`、`if` 短语句、`switch`（无 `break`）。
- 表驱动测试里多返回值要一起断言，遇到错误时先判断 `err` 再看结果。
- 多值返回 + `go test ./...` 是默认开发流程，肌肉记忆。
