# Go 接口 — 问题版（自测用）

## Q1

Go 的接口和 C++ 的抽象类 / Python 的 ABC（Abstract Base Class / 抽象基类）有什么区别？

<details>
<summary>答案</summary>

**Go 接口是隐式实现**（structural typing），其他语言是显式实现（nominal typing）：

```go
// Go — 鸭子类型
type Writer interface { Write([]byte) (int, error) }
type MyWriter struct {}
func (m MyWriter) Write(b []byte) (int, error) { return len(b), nil }
// MyWriter 自动实现了 Writer，不需要写 "implements"
```

```cpp
// C++ — 显式继承
class Writer { public: virtual int write(const char*) = 0; };
class MyWriter : public Writer { ... };  // 必须写 : public Writer
```

```python
# Python — 显式注册
class Writer(ABC):  # ABC = Abstract Base Class
    @abstractmethod
    def write(self, data): ...
class MyWriter(Writer): ...  # 必须继承 Writer
```

Go 的优势：可以给**已经存在的类型**实现接口（只要定义在同一个包）。

</details>

## Q2

接口在 Go 里的底层表示是什么？（`interface` 的值是怎么存的？）

<details>
<summary>答案</summary>

接口值底层是一个**二元组**（type, value），类似：

```go
type iface struct {
    tab  *itab        // 类型信息 + 方法表
    data unsafe.Pointer  // 实际数据的指针
}
```

```go
var w Writer
w = MyWriter{}
// w = {tab: &{type: MyWriter, methods: [...]}, data: &MyWriter{}}
w = nil
// w = {tab: nil, data: nil}
```

这就是为什么 `var x any = 42` 后 `x == nil` 是 false——type 不为空。

```go
var p *MyWriter = nil
var w Writer = p
w == nil  // ❌ false！因为 w.tab != nil
```

</details>

## Q3

空接口 `interface{}` 或 `any` 有什么用？什么时候该用？

<details>
<summary>答案</summary>

空接口可以持有任何类型的值。类似于 C++ 的 `void*` 或 Python 的 `object`，但**类型安全**。

什么时候用：
- 需要处理未知类型（如 `fmt.Println(any)`）
- JSON 解析不确定结构（`map[string]any`）
- 通用容器（少见）

```go
// JSON 任意解析
var data map[string]any
json.Unmarshal([]byte(`{"name": "hello", "count": 42}`), &data)
name := data["name"].(string)    // 需要类型断言
count := data["count"].(float64) // JSON 数字解析为 float64
```

**不要滥用**：能用具体类型就用具体类型。用 `any` 等于放弃了编译期类型检查。

</details>

## Q4

errors.Is 和 errors.As 是做什么的？

<details>
<summary>答案</summary>

```go
// errors.Is — 检查错误链中是否有特定错误值
if errors.Is(err, io.EOF) {
    // 是 EOF（可以是包装过的 EOF）
}

// errors.As — 从错误链中找到匹配类型的错误
var pathError *os.PathError
if errors.As(err, &pathError) {
    fmt.Println("path:", pathError.Path)
}
```

- `Is` 比较**值**（常用 sentinel error 如 `io.EOF`）
- `As` 匹配**类型**（把错误转换成具体类型来获取更多信息）

两者都会遍历错误链（通过 `Unwrap()`），类似于 C++ 异常链的 catch 匹配。

</details>
