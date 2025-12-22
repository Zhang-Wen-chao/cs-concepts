# 04 · sync 与 atomic 工具

> 资料：`sync` 包文档、Go Memory Model、articles by Go Team on Mutex performance。

## Mutex 家族
- `sync.Mutex`：独占锁；使用 `defer mu.Unlock()` 最安全，但注意热点路径性能。
- `sync.RWMutex`：读写锁；读多写少时提升吞吐；写锁与读锁互斥，避免长时间持有。
- `sync.Once`：保证函数只执行一次，常用于懒加载。

## 协调工具
- `sync.WaitGroup`：统计 goroutine 完成情况，`Add`/`Done` 必须配对；不要复制 `WaitGroup`。
- `sync.Cond`：条件变量，与 `Mutex` 搭配实现复杂同步；现代代码更多用 channel 代替。
- `sync.Map`：适合读多写少且 key 不可预测的场景（如缓存、监控标签）；不等同于标准 map。

## atomic
- `atomic.AddInt64`、`atomic.Value` 等提供 lock-free 操作；需 import `sync/atomic`。
- 使用 atomic 时要阅读内存模型，理解 happens-before，避免“写后读”未同步的问题。

## 决策建议
| 场景 | 推荐 |
| --- | --- |
| 控制 goroutine 结束 | `context` + channel |
| 累计计数/状态 | atomic（轻量）或 `sync.Mutex` |
| 缓存/Map | `map + Mutex`（可分段）或 `sync.Map` |
| 初始化单例 | `sync.Once` |

## Checklist
- [ ] 区分数据竞争 vs 逻辑竞争，知道何时用 Mutex/atomic/channel。
- [ ] WaitGroup + goroutine 一对一配对，不遗漏 `Done`。
- [ ] 在 `go test -race` 下验证所有共享数据均受保护。
