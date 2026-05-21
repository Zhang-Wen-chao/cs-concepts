# 03 · 集合与引用语义

> 资料：Go Blog「Go Slices: usage and internals」、Effective Go 数组/切片章节、Go Tour `moretypes` 模块。

## 数组 vs 切片
- **数组**：`[N]T`，长度是类型一部分，赋值/传参会复制整个数组。
- **切片**：`[]T`，包含指向底层数组的指针、长度、容量，按引用语义传递。
- `len` 返回元素个数，`cap` 返回容量；`append` 超出容量时会分配新底层数组。

```go
names := []string{"go", "cpp"}
copyNames := names       // 共享底层数组
clone := append([]string(nil), names...) // 独立副本
```

## map 语义
- `map[K]V` 默认零值为 `nil`，必须 `make(map[K]V)` 才能写入。
- 读不存在的键返回零值，可结合 `v, ok := m[k]` 判断是否存在。
- 遍历顺序不稳定，需要稳定顺序时先收集 key 并排序。

## 常见操作
- `copy(dst, src)`：按最小长度复制；适合切片备份。
- `append(dst, src...)`：`...` 可展开切片。
- `delete(m, key)`：从 map 中移除键。

## 建议
1. 传递只读切片时用 `[]T`，写操作前 `clone := append([]T(nil), src...)`。
2. map 作为函数参数时无需返回（引用语义），除非要返回错误或新 map。
3. 通过 `var zero []T` 表示“未初始化但可读”的切片；用 `make([]T, 0, n)` 预分配。

## Checklist
- [ ] 能解释数组/切片差异，并说明 `append` 什么时候会重新分配。
- [ ] map 零值/存在判断/删除操作熟练掌握。
- [ ] 写过至少一个“切分/聚合”函数并配有表驱动测试。

> 参考答案
> 1. 数组长度固定，赋值/传参会整块复制；切片由指针+长度+容量组成，按引用语义传递，`append` 在超出 `cap` 时会分配新底层数组。
> 2. 零值 `nil` map 只能读不能写；`v, ok := m[k]` 判断存在；`delete(m, key)` 无论 key 是否存在都安全。
> 3. 示例：`playground/03_collections` 中的 `Chunk`/`MergeCounters`，配有表驱动测试覆盖正常、空输入与错误分支。
