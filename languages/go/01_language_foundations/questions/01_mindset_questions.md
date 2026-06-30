# Go 思维 — 问题版（自测用）

> 盖住答案，能说上来就算过关。

## Q1

Go 和 C++/Python 在并发模型上最大的区别是什么？

<details>
<summary>答案</summary>

| 语言 | 并发模型 |
|---|---|
| Go | goroutine（M:N 调度，轻量级用户态线程） |
| C++ | `std::thread`（1:1 对应 OS 线程，重量级） |
| Python | 线程（受 GIL 限制）+ asyncio（协作式） |

Go 的 goroutine 栈初始只有几 KB，可以轻松开上万甚至百万个。C++ 的线程栈默认几 MB，开多了就崩。

</details>

## Q2

`var x int = 10` 和 `x := 10` 有什么区别？

<details>
<summary>答案</summary>

- `var x int = 10` — 完整声明，显式指定类型
- `x := 10` — 短变量声明，编译器推导类型（x 是 int）

短声明只能在函数内部用，不能在包级别用。而且 `:=` **至少有一个新变量**才能用。

</details>

## Q3

Go 的零值（zero value）机制是什么？C++ 呢？

<details>
<summary>答案</summary>

Go：每个变量声明后自动初始化为零值，不需要手动初始化。
- `int` → `0`，`string` → `""`，`bool` → `false`，`*T` → `nil`，`struct` 每个字段递归零值

```go
var x int      // x = 0
var s string   // s = ""
```

C++：**没有零值**，不初始化就是未定义行为。

```cpp
int x;   // x 的值不确定 — undefined behavior
```

</details>

## Q4

Go 的 `error` 是接口，为什么不用异常（try/catch）？

<details>
<summary>答案</summary>

Go 的设计哲学是**显式错误处理**。错误是普通的值，不是控制流。

```go
f, err := os.Open("file.txt")
if err != nil {
    return fmt.Errorf("open failed: %w", err)
}
```

C++/Python 用异常的一个问题是：你不知道哪个函数会抛出异常，不 catch 就崩。
Go 强迫你每步都处理错误——烦，但安全。你不会意外忽略错误。

</details>

## Q5

Go 的 `interface{}`（或 `any`）和 C++ 的 `void*` / Python 的 `Object` 有什么区别？

<details>
<summary>答案</summary>

- Go `any`（= `interface{}`）：**类型安全的**。存了值 + 类型信息，取出时需要类型断言，运行时检查
- C++ `void*`：**完全不安全的**。丢掉所有类型信息，用之前得 cast，写错就 UB
- Python `Object`：一切皆对象，天生动态，不需要"装箱"

```go
var x any = 42
v := x.(int)         // 类型断言，运行时检查
s := x.(string)      // panic！
```

</details>

## Q6

Go 里 `make` 和 `new` 有什么区别？

<details>
<summary>答案</summary>

- `new(T)` — 分配内存，返回 `*T`（指向零值 T 的指针）。用于值类型（struct，int 等）
- `make(T)` — 只用于 slice/map/channel，返回初始化的**非零值**（不是指针）

```go
p := new(int)        // *int，指向 0
s := make([]int, 5)  // []int{0,0,0,0,0}
m := make(map[int]string) // 空的 map，可以直接用
```

`new` 在 Go 里很少用，大部分时候 `&T{}` 就够了。
