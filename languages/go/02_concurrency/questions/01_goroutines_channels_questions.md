# Go 并发 — 问题版（自测用）

## Q1

什么是 goroutine 泄漏？怎么查？

<details>
<summary>答案</summary>

Goroutine 启动后一直阻塞，永远无法退出，就叫泄漏。常见原因：
- 向无缓冲 channel 发送，但没有接收方
- 从 channel 接收，但没有发送方
- `select` 里所有 case 都永远阻塞
- 无限循环且没有退出条件

**怎么查：**
```go
// 获取当前 goroutine 数量
fmt.Println(runtime.NumGoroutine())

// pprof 查看 goroutine 栈
import _ "net/http/pprof"
// 访问 /debug/pprof/goroutine
```

</details>

## Q2

无缓冲 channel 和有缓冲 channel 的区别？

<details>
<summary>答案</summary>

```go
ch := make(chan int)      // 无缓冲：发送和接收必须同时准备好
ch := make(chan int, 5)   // 有缓冲：缓冲区有空间就可以发送
```

- **无缓冲**：同步通信。发送方阻塞直到接收方取走，接收方阻塞直到有数据来
- **有缓冲**：异步通信。发送方在缓冲区满之前不阻塞，接收方在缓冲区空之前不阻塞

无缓冲 channel 常用于**协程同步**（确保 goroutine A 在 B 之前完成某件事）。
有缓冲 channel 常用于**限流/队列**。

</details>

## Q3

`close(ch)` 之后还能从 channel 读数据吗？

<details>
<summary>答案</summary>

可以。关闭后 channel 里剩余的数据仍然可以被读到：

```go
ch := make(chan int, 3)
ch <- 1; ch <- 2; close(ch)

for v := range ch {
    fmt.Println(v)  // 1, 2 — 读完为止
}
v, ok := <-ch       // v=0, ok=false — channel 已空且已关闭
```

- 关闭后读空 channel 返回零值 + `ok=false`
- 向已关闭的 channel 发送 → **panic**
- 重复关闭 → **panic**
- 关闭 nil channel → **panic**

实践中：通常是发送方负责关闭 channel，接收方不应该关闭。

</details>

## Q4

Context 的三种派生方式和各自的用途？

<details>
<summary>答案</summary>

```go
// 1. 超时取消
ctx, cancel := context.WithTimeout(parent, 5*time.Second)
defer cancel()

// 2. 手动取消
ctx, cancel := context.WithCancel(parent)
defer cancel()

// 3. 截止时间
ctx, cancel := context.WithDeadline(parent, time.Now().Add(5*time.Second))
defer cancel()
```

用途：
- **WithTimeout**：外部请求超时控制（最常见）
- **WithCancel**：手动取消 goroutine（用户点了取消按钮）
- **WithDeadline**：固定时间点截止（缓存过期）

原则：**Context 要作为函数第一个参数传递**，不要存到 struct 里。

</details>

## Q5

`sync.WaitGroup` 和 `errgroup.Group` 有什么区别？

<details>
<summary>答案</summary>

```go
// sync.WaitGroup — 只等完成，不关心错误
var wg sync.WaitGroup
for i := 0; i < 5; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        // 干活...
    }()
}
wg.Wait()

// errgroup.Group — 等完成 + 收集第一个错误
g, ctx := errgroup.WithContext(ctx)
for _, task := range tasks {
    task := task
    g.Go(func() error {
        return doTask(ctx, task)
    })
}
if err := g.Wait(); err != nil {
    // 第一个出错的 goroutine 的错误
}
```

- `WaitGroup`：简单等待，不关心错误
- `errgroup`：等待 + 收集错误 + 第一个错误自动取消其他 goroutine（通过 context）

</details>

## Q6

`sync.Mutex` 和 `sync.RWMutex` 什么时候用哪个？

<details>
<summary>答案</summary>

- `sync.Mutex`：读写互斥，一次只能一个人用
- `sync.RWMutex`：读共享，写互斥

```go
var mu sync.RWMutex
var data map[string]int

// 写操作 — 排他锁
mu.Lock()
data["key"] = 42
mu.Unlock()

// 读操作 — 共享锁（多个读者可以同时读）
mu.RLock()
v := data["key"]
mu.RUnlock()
```

**原则**：
- 读远多于写 → `RWMutex`
- 读写差不多或写多 → `Mutex`（RWMutex 的 RLock/RUnlock 比 Lock/Unlock 略慢）

</details>
