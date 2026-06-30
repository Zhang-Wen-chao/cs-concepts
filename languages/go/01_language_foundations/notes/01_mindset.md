# Go 思维方式

> Go 不是 Python，也不是 C++。它的设计哲学非常独特。

## 核心哲学

### 1. 少即是多

Go 只有 **25 个关键字**（C++ 有 80+，Python 有 35+）。没有：
- ❌ 类继承（用组合）
- ❌ 泛型（Go 1.18 之前；现在的泛型也有限制）
- ❌ 异常（用 error 值）
- ❌ 运算符重载
- ❌ 继承（vs C++）

少 -> 代码更容易理解 -> 更容易维护。

### 2. 显式优于隐式

```go
// Go — 错误必须显式处理
f, err := os.Open("file.txt")
if err != nil {
    // 你没法忽略 err
}

// Python — 可以忽略异常
f = open("file.txt")  # 不 try 也不管
```

### 3. 并发是内置的

```go
go doWork()   // 启动一个 goroutine，比启动线程轻量 1000 倍
```

C++ 的线程是标准库扩展，Python 的线程有 GIL 限制。Go 的 goroutine 从一开始就是语言的一部分。

## 大写 = 公开

Go 用**大小写**控制可见性，不是 `public/private`：

```go
func PrivateFunc() {}   // 大写字母开头 = 可导出（公开）
func privateFunc() {}   // 小写字母开头 = 不可导出（私有）
```

简洁而没有歧义。

## 错误不是异常

```go
// Go 风格：错误是值
result, err := doSomething()
if err != nil {
    // 处理错误，不是抛异常
    return fmt.Errorf("doSomething failed: %w", err)
}
```

对比：
- Python：`raise Exception("...")` — 不 catch 就崩
- C++：`throw std::runtime_error(...)` — 不 catch 就崩
- Go：`return err` — 你必须检查它

## 零值哲学

```go
var x int         // x = 0
var s string      // s = ""
var p *int        // p = nil
```

不初始化也有安全值，这是 Go 的**零值保证**。对比 C++ 不初始化就是 UB。

## 总结

| 特性 | Go | Python | C++ |
|------|-----|--------|-----|
| 类型系统 | 静态，强类型 | 动态，强类型 | 静态，强类型 |
| 并发 | goroutine + channel | GIL + asyncio | std::thread |
| 错误 | error 值 | 异常 | 异常 |
| 公开/私有 | 大写/小写 | \_ 约定 | public/private |
| 零值 | 有 | N/A | 无（UB） |
