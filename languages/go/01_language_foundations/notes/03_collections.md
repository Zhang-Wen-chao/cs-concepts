# Go 复合类型

## Array（数组）

```go
var arr [5]int             // [0 0 0 0 0]
arr := [5]int{1, 2, 3, 4, 5}
arr := [...]int{1, 2, 3}   // 由编译器推导长度
```

数组是**值类型**——传参会拷贝全部元素。这点跟 C++ 的内置数组不同（C++ 数组传参退化为指针）。

## Slice（切片）

```go
var s []int                 // nil slice: len=0, cap=0
s := []int{1, 2, 3}         // len=3, cap=3
s := make([]int, 5)         // len=5, cap=5, 全零
s := make([]int, 0, 10)     // len=0, cap=10

// 子切片（共享底层数组）
arr := [5]int{0, 1, 2, 3, 4}
s := arr[1:4]               // [1, 2, 3]
```

Slice 是 Go 中最常用的"动态数组"。底层结构：

```
┌──────────┐
│  ptr ──────→ [底层数组]
│  len = 3  │
│  cap = 5  │
└──────────┘
```

### append

```go
var s []int
s = append(s, 1)        // [1]
s = append(s, 2, 3)     // [1, 2, 3]
s = append(s, []int{4,5}...)  // [1, 2, 3, 4, 5]
```

`append` 在 cap 不够时分配新数组（通常翻倍增长），返回新 slice。**必须用返回值。**

```go
s = append(s, 1)   // ✅
append(s, 1)       // ❌ 没用——返回值丢了
```

### copy

```go
src := []int{1, 2, 3}
dst := make([]int, len(src))
n := copy(dst, src)  // n = 3，dst = [1, 2, 3]
```

## Map

```go
var m map[string]int              // nil map，不能写入
m := make(map[string]int)         // ✅ 空 map，可以写入
m := map[string]int{"a": 1, "b": 2}

// CRUD
m["c"] = 3                         // 写入
v := m["a"]                        // 读取（key 不存在返回零值）
v, ok := m["z"]                    // v=0, ok=false（检查 key 是否存在）
delete(m, "a")                     // 删除
```

Map 遍历顺序**随机**——不要依赖顺序。

## Struct

```go
type Person struct {
    Name string
    Age  int
}

// 创建
p := Person{Name: "Alice", Age: 30}
p := Person{"Alice", 30}           // 按字段顺序初始化（不推荐）

// 访问
fmt.Println(p.Name)                // Alice

// 方法（在 struct 外部定义）
func (p Person) Greet() string {   // 值接收者
    return "Hello, I'm " + p.Name
}

func (p *Person) SetAge(age int) { // 指针接收者
    p.Age = age
}
```

Go 的方法定义在 struct **外面**——这是 Go 和 C++/Python 的重要区别。C++ 的方法写在 class 内部，Python 也是。

## 比较

| 类型 | `==` 可比较？ | 作为 map key？ |
|------|:---:|:---:|
| bool | ✅ | ✅ |
| 数值 | ✅ | ✅ |
| string | ✅ | ✅ |
| 指针 | ✅ | ✅ |
| struct（全是可比较字段） | ✅ | ✅ |
| struct（含 slice/map） | ❌ | ❌ |
| slice | ❌ | ❌ |
| map | ❌ | ❌ |
