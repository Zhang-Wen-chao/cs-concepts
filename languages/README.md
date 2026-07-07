# Multi-Language Learning

> 同时学 C++、Go、Python 三门语言，每门都是二当家的标配。

## 🗺️ 概念地图

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

## 📂 目录结构

```
languages/
├── README.md              ← 你在这
├── cpp/                   ← C++ 8 Stage
│   ├── 00_compilation_fundamentals/  笔记 3 篇
│   ├── 01_cpp_vs_python/            自测 8 套
│   ├── 02_memory_and_raii/          笔记 3 篇
│   ├── 03_oop_deep_dive/            笔记 4 篇 ✅
│   ├── 04_templates/                 笔记 4 篇 ✅
│   ├── 05_stl_internals/            笔记 4 篇 ✅
│   ├── 06_modern_cpp/               笔记 5 篇 ✅
│   ├── 07_concurrency/              笔记 5 篇 ✅
│   └── _reference/                  已完成的老内容
├── go/                     ← Go 4 阶段
│   ├── 01_language_foundations/     笔记 + 自测 4 套
│   ├── 02_concurrency/              笔记 + 自测 1 套
│   ├── 03_engineering/              笔记 + 自测 1 套
│   └── 04_projects/                 项目代码
└── python/                 ← Python 4 阶段 ✅
    ├── 01_foundations/              笔记 6 篇 + 自测 1 套 + playground
    ├── 02_intermediate/             笔记 7 篇 + playground（含 async）
    ├── 03_engineering/              笔记 4 篇 + playground
    └── 04_projects/                 日志分析 CLI + API 服务（14 测试）
```

## 🧭 当前进度

| 语言 | 当前阶段 | 路径 |
|---|---|---|---|
| → C++ | Stage 1 ⬅️ **正在学** | `cpp/README.md` |
| → Go | 阶段 2 并发 / 已有内容复习 | `go/README.md` |
| → Python | 02_intermediate + 04_projects ✅ | `python/README.md` |
