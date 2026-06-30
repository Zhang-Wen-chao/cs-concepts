# C++ 学习路径

> 从零基础到能讲清机制、写出现代 C++、从容应对 C++ 面试

## 🗺️ 8 阶段路线图

| Stage | 主题 | 核心问题 | 目录 | 笔记数 |
|---|---|---|---|---|
| 0 | 编译底层 | 编译器到底做了什么？ | `00_compilation_fundamentals/` | 3 篇 |
| 1 | C++ vs Python 本质 | 静态类型、值语义、栈堆是什么？ | `01_cpp_vs_python/` | 自测 8 套 |
| 2 | 内存模型与 RAII | 谁负责释放内存？move 是什么？ | `02_memory_and_raii/` | 3 篇 |
| 3 | 面向对象深入 | 虚函数表怎么工作？ | `03_oop_deep_dive/` | 4 篇 |
| 4 | 模板与泛型 | 模板怎么实例化？ | `04_templates/` | 4 篇 |
| 5 | STL 源码级 | vector / unordered_map 怎么实现？ | `05_stl_internals/` | 4 篇 |
| 6 | 现代 C++ 特性 | C++11/14/17/20 怎么用对？ | `06_modern_cpp/` | 5 篇 |
| 7 | 并发与内存序 | 内存序是怎么回事？ | `07_concurrency/` | 5 篇 |

## ✅ 进度

- [x] Stage 0 — 编译底层
- [ ] Stage 1 — C++ vs Python 本质
- [ ] Stage 2 — 内存模型与 RAII
- [x] Stage 3 — 面向对象深入（笔记到位）
- [x] Stage 4 — 模板与泛型（笔记到位）
- [x] Stage 5 — STL 源码级（笔记到位）
- [x] Stage 6 — 现代 C++ 特性（笔记到位）
- [x] Stage 7 — 并发与内存序（笔记到位）

**注意**：Stages 0-2 放在 01_cpp_vs_python/ 和 02_memory_and_raii/ 下，内容已具雏形但自测通过后才真正勾上。

## 📖 完成的老内容

`_reference/` 下存放旧学习路径的完成内容，需要时翻阅。

- `_reference/01_modern_cpp_basics/` — RAII、智能指针、Lambda、模板基础
- `_reference/02_concurrency/` — 线程基础、锁、条件变量、线程池
- `_reference/03_system_programming/` — 网络 I/O、序列化、性能优化
- `_reference/04_projects/` — 简易 RPC、brpc、推荐服务
- `_reference/multithreading/` — 多线程练习
