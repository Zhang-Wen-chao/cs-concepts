# 04 · 组合与接口

> 资料：Go Tour「Methods & Interfaces」、Effective Go「Methods」「Interfaces and other types」。

## struct & 嵌入
- `type Server struct { Host string; Port int }`：字段大写即导出。
- 匿名字段（嵌入）用于组合：`type Service struct { *log.Logger }`，可直接访问嵌入类型的方法。
- `new(T)` / `&T{}` 返回指针；对 struct 方法使用值/指针接收者需考虑是否修改状态。

## 方法集
- 值类型 `T` 的方法集：接收者为 `T` 的方法。
- 指针类型 `*T` 的方法集：包含接收者为 `T` 和 `*T` 的方法。
- 接口满足条件：类型实现接口的所有方法即可（鸭子类型）。

## 接口设计
- 小而专：通常定义 1~3 个方法，比如 `io.Reader`、`http.Handler`。
- 约定优于继承：结构体组合 + 接口约束比 class 继承更灵活。
- 接口引入点尽量靠近使用处，避免过度抽象。

## 类型断言/切换
```go
if c, ok := shape.(Circle); ok {
    return c.Radius
}

switch v := anyShape.(type) {
case Circle:
    // ...
default:
}
```

## Checklist
- [ ] 编写过带嵌入字段的 struct，并能解释方法集规则。
- [ ] 理解 "接口即契约"：消费端根据需要定义接口，而非在生产端预先设计。
- [ ] 通过 type assertion/switch 处理多态行为并写测试覆盖。
