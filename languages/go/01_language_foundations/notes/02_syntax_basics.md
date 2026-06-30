# Go 语法基础

## 变量声明

```go
// 四种声明方式
var x int = 10       // 完整声明
var x = 10           // 类型推导
x := 10              // 短声明（只能在函数内）
var x int            // 零值声明（x = 0）
```

**短声明 `:=`** 是最常用的。它声明并初始化，编译器推导类型。

## 基本类型

```go
bool                    // true/false
int int8 int16 int32 int64
uint uint8 uint16 uint32 uint64
uintptr                 // 指针类型，大小随平台
float32 float64
complex64 complex128    // 复数！
byte                    // = uint8
rune                    // = int32（Unicode code point）
string                  // UTF-8，不可变
```

**没有 `char` 类型**。C++ 的 `char` 在 Go 里用 `byte` 或 `rune`。

## 控制流

### if

```go
if x > 0 {
    fmt.Println("positive")
} else if x < 0 {
    fmt.Println("negative")
} else {
    fmt.Println("zero")
}

// if 可以跟一个语句
if err := doSomething(); err != nil {
    fmt.Println("error:", err)
}
// err 的作用域只在 if 块内
```

### for（唯一循环关键字）

```go
// 传统
for i := 0; i < 10; i++ { }

// while 风格
for x < 10 { x++ }

// 无限循环
for { break }

// range
for i, v := range []int{1, 2, 3} { }
for k, v := range map[string]int{"a": 1} { }
```

**只有 `for`，没有 `while` 和 `do-while`。**

### switch

```go
switch x {
case 1:
    fmt.Println("one")  // 自动 break，不穿透
case 2, 3:
    fmt.Println("two or three")
default:
    fmt.Println("other")
}

// switch 可以没有表达式（当 if 用）
switch {
case x < 0:
    fmt.Println("negative")
case x == 0:
    fmt.Println("zero")
}
```

## 函数

```go
// 多返回值
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// 命名返回值
func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return  // "裸返回"——返回当前 x, y 的值
}
```

**多返回值**是 Go 函数的核心特性。C++ 用输出参数或 tuple，Python 用 tuple。

## defer

```go
func readFile(name string) error {
    f, err := os.Open(name)
    if err != nil { return err }
    defer f.Close()     // 函数返回前执行

    // 处理文件...
    return nil
}
```

`defer` 确保资源释放，是 Go 的 RAII 等价物。多个 defer **后进先出**。

## 指针

```go
var x int = 42
var p *int = &x    // 指向 x
*p = 21             // 修改 x
```

**没有指针运算**（不像 C++ 可以 `p++`）。

## 总结

| 特性 | Go 写法 | C++ 写法 | Python 写法 |
|------|---------|----------|-------------|
| 变量 | `x := 10` | `int x = 10;` | `x = 10` |
| 范围 for | `for i, v := range` | `for (auto& v : vec)` | `for i, v in enumerate()` |
| 错误返回 | `(T, error)` | 异常/输出参数 | 异常 |
| 释放资源 | `defer f.Close()` | RAII/析构 | `with` / `finally` |
