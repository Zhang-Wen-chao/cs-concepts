# Go 语法基础 — 问题版（自测用）

## Q1

Go 的 `defer` 是做什么的？多个 `defer` 的执行顺序？

<details>
<summary>答案</summary>

`defer` 延迟执行一个函数调用，在包含它的函数**返回前**执行。用于资源清理（关闭文件、解锁等）。

```go
f, _ := os.Open("file.txt")
defer f.Close()  // 函数结束时自动关闭
```

多个 `defer` **后进先出**（LIFO / 栈顺序）：

```go
defer fmt.Println("first")   // 第三个执行
defer fmt.Println("second")  // 第二个执行
defer fmt.Println("third")   // 第一个执行（最后 defer 的先执行）
```

</details>

## Q2

Go 的 slice 和 array 有什么区别？slice 的底层结构是什么？

<details>
<summary>答案</summary>

- **array**：固定长度，值类型。`[5]int` 和 `[10]int` 是不同类型
- **slice**：动态长度，引用类型。`[]int` 是切片

slice 底层结构：

```go
type slice struct {
    ptr  unsafe.Pointer  // 指向底层数组的指针
    len  int             // 当前元素个数
    cap  int             // 底层数组容量
}
```

slice 作为参数传递时，拷贝的是这个结构体（ptr/len/cap），**但底层数组是共享的**。修改 slice 的元素会改到调用方。

</details>

## Q3

Go 的 `map` 遍历顺序是确定的吗？

<details>
<summary>答案</summary>

**不是。** Go 故意**随机化** map 的遍历顺序。每次遍历顺序都可能不同。

```go
m := map[string]int{"a": 1, "b": 2, "c": 3}
for k, v := range m {
    fmt.Println(k, v)  // 每次执行顺序可能不一样
}
```

这是 Go 设计者故意的——防止你依赖 map 顺序。如果你需要有序遍历，先取出 key 排序。

</details>

## Q4

Go 的 `switch` 和 C++ 的 `switch` 有什么关键区别？

<details>
<summary>答案</summary>

```go
// Go — 不用 break，自动跳出
switch x {
case 1:
    fmt.Println("one")
    // 不用 break，不会 fall through
case 2:
    fmt.Println("two")
}
```

区别：
1. Go 的每个 case **自动 break**，不会穿透到下个 case
2. 想要穿透要显式写 `fallthrough`（很少用）
3. Go 的 case 可以跟多个值：`case 1, 2, 3:`
4. Go switch 可以**没有表达式**，当成 if/else if 链用

```cpp
// C++ — 必须手动 break，忘了就顺序执行！
switch (x) {
    case 1:
        std::cout << "one";
        // break; 忘了写！继续执行 case 2
    case 2:
        std::cout << "two";
}
```

</details>

## Q5

Go 的 `goroutine` 和 `thread` 是什么关系？M:N 调度是什么？

<details>
<summary>答案</summary>

- **goroutine**：Go 的用户态"协程"，轻量级（初始栈 2-4KB）
- **OS 线程**：内核态，重量级（默认栈 1-8MB）

**M:N 调度** = M 个 goroutine 跑在 N 个 OS 线程上。Go 运行时的调度器负责分配：

```
goroutine  →→→→→→  Go 调度器  →→→→→→  OS 线程  →→→→→→  CPU 核心
  (成千上万)           (M:N)         (几十个)        (几个)
```

好处：
- 创建 goroutine 很快（~1μs vs 线程 ~10μs 到 ms 级）
- 切换成本低（用户态，不涉及内核态切换）
- 可以开几万个 goroutine

</details>

## Q6

Go 的 `select` 语句是做什么的？

<details>
<summary>答案</summary>

`select` 同时等待多个 channel 操作，哪个先准备好就执行哪个：

```go
select {
case msg := <-ch1:
    fmt.Println("got from ch1:", msg)
case ch2 <- 42:
    fmt.Println("sent to ch2")
case <-time.After(1 * time.Second):
    fmt.Println("timeout!")
default:
    fmt.Println("nobody ready")
}
```

- 多个 case 同时准备好时，**随机选一个**执行
- `default` 在所有 channel 都阻塞时执行（非阻塞 select）
- 常用于超时、多路复用、优雅退出

这是 Go 独有的并发原语，C++ 和 Python 没有直接等价物。
