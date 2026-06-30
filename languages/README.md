# Multi-Language Learning

> 同时学 C++、Go、Python 三门语言，每门都是二当家的标配。

## 🗺️ 概念地图

同一个概念，三门语言怎么做。

| 概念 | C++ | Go | Python |
|---|---|---|---|
| **类型** | 静态，值语义 | 静态，结构体 | 动态，一切皆对象 |
| **内存** | 手动/RAII/智能指针 | GC（并发标记清扫） | GC（引用计数） |
| **并发** | `std::thread` / 锁 | goroutine + channel | 线程/GIL，asyncio |
| **错误处理** | 异常 `try/catch` | 多返回值 `error` | 异常 `try/except` |
| **接口/多态** | 虚函数 / 模板 | interface 隐式实现 | 鸭子类型 / ABC |
| **包管理** | 无官方（CMake） | `go mod` | `pip` / `poetry` |
| **零值** | 必须初始化 | 自动零值 | None |
| **所有权** | unique_ptr / 左值右值 | 无（GC） | 无（GC） |

## 🧭 当前进度

跳到对应目录看详细进度：

| 语言 | 当前阶段 | 路径 |
|---|---|---|
| → C++ | Stage 1 ⬅️ **正在学** | `cpp/README.md` |
| → Go | 阶段 2 并发 / 已有内容复习 | `go/README.md` |
| → Python | 刚起步 | `python/README.md` |

## 🎯 学习口号

> **先知其然，再知其所以然。**
> 会用只是门槛，能讲清楚机制才算会。
> 每个概念，想想"另两门语言怎么做"。
