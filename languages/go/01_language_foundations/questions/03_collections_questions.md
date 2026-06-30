# Go 复合类型 — 问题版（自测用）

## Q1

`[5]int` 和 `[]int` 传给函数时，行为有什么不同？

<details>
<summary>答案</summary>

```go
func foo(arr [5]int) { arr[0] = 999 }   // 拷贝整个数组，外面不变
func foo(sl []int) { sl[0] = 999 }      // 共享底层数组，外面也变！
```

- array 是**值语义**：传参拷贝全部元素
- slice 是**引用语义**（内部）：传参拷贝 struct（ptr/len/cap），但 ptr 指向同一个底层数组

但注意：`append` 如果超过 cap 会分配新数组，原 slice 不受影响。

</details>

## Q2

Go 的 `nil slice` 和 `empty slice` 有什么区别？

<details>
<summary>答案</summary>

```go
var s []int           // nil slice: ptr=nil, len=0, cap=0
s = []int{}           // empty slice: ptr=有效地址, len=0, cap=0
s = make([]int, 0)    // empty slice: ptr=有效地址, len=0, cap=0
```

**实践中的区别**：
- `nil slice`：json 序列化为 `null`
- `empty slice`：json 序列化为 `[]`
- `nil slice` 可以 `append`（自动分配底层数组）
- `len(nil slice)` = 0，`range nil slice` 不会迭代

通常用 `var s []T` 声明（nil slice），除非你确定要 JSON 输出 `[]`。

</details>

## Q3

Go 的 `for range` 遍历 slice 或 map 时，元素是拷贝还是引用？

<details>
<summary>答案</summary>

**值是拷贝。** 每次迭代都拷贝元素到循环变量。

```go
items := []MyStruct{{Name: "a"}, {Name: "b"}}
for _, item := range items {
    item.Name = "xxx"  // ❌ 改的是拷贝，不是原 slice 的元素
}
// items[0].Name 还是 "a"
```

要改原 slice 的元素：
```go
for i := range items {
    items[i].Name = "xxx"   // ✅
}
// 或者用指针 slice
for _, item := range items {
    item.Name = "xxx"   // 如果 items 是 []*MyStruct 就可以
}
```

</details>

## Q4

Go 的 `struct` 比较有什么限制？什么时候可以比较？

<details>
<summary>答案</summary>

struct 可以用 `==` 比较的前提：**所有字段都是可比较的**（基本类型、指针、可比较的 struct）。

```go
type A struct { X int; Y string }
a1, a2 := A{1, "a"}, A{1, "a"}
a1 == a2  // ✅ true

type B struct { X int; S []int }
b1, b2 := B{1, []int{1}}, B{1, []int{1}}
b1 == b2  // ❌ 编译错误：slice 不可比较
```

不能比较的 struct 不能作为 map 的 key。

</details>

## Q5

Go 的 `map` 并发安全吗？怎么安全地并发读写？

<details>
<summary>答案</summary>

**不安全。** 并发读写 map 会导致 **fatal error: concurrent map read and map write**，直接崩溃。

安全方案：

1. **`sync.RWMutex`** — 读写锁
```go
var mu sync.RWMutex
m := make(map[string]int)

// 写
mu.Lock()
m["key"] = 42
mu.Unlock()

// 读
mu.RLock()
v := m["key"]
mu.RUnlock()
```

2. **`sync.Map`** — 标准库的并发安全 map（适合读多写少的场景）

3. **channel 序列化访问** — Go 哲学：通过通信共享内存

</details>
