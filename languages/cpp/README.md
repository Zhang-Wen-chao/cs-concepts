# C++ 学习路径

> 从零基础到能讲清机制、写出现代 C++、从容应对 C++ 面试

## 🗺️ 8 阶段路线图

| Stage | 主题 | 核心问题 | 目录 |
|---|---|---|---|
| 0 | 编译底层 | 编译器到底做了什么？ | `00_compilation_fundamentals/` |
| 1 | C++ vs Python 本质 | 静态类型、值语义、栈堆是什么？ | `01_cpp_vs_python/` |
| 2 | 内存模型与 RAII | 谁负责释放内存？move 是什么？ | `02_memory_and_raii/` |
| 3 | 面向对象深入 | 虚函数表怎么工作？ | (待建) |
| 4 | 模板与泛型 | 模板怎么实例化？ | (待建) |
| 5 | STL 源码级 | vector / unordered_map 怎么实现？ | (待建) |
| 6 | 现代 C++ 特性 | C++11/14/17/20 怎么用对？ | (待建) |
| 7 | 并发与内存序 | 内存序是怎么回事？ | (待建) |

## ✅ 进度

- [x] Stage 0 — 编译底层
- [ ] Stage 1 — C++ vs Python 本质
- [ ] Stage 2 — 内存模型与 RAII
- [ ] Stage 3 — 面向对象深入
- [ ] Stage 4 — 模板与泛型
- [ ] Stage 5 — STL 源码级
- [ ] Stage 6 — 现代 C++ 特性
- [ ] Stage 7 — 并发与内存序

### Stage 1 细化

- [x] 1.7 智能指针
- [ ] 1.1 静态类型
- [ ] 1.2 编译执行
- [ ] 1.3 内存管理
- [ ] 1.4 值语义
- [ ] 1.5 错误处理
- [ ] 1.6 RAII
- [ ] 1.8 头文件 & ODR
- [ ] 1.9 未定义行为 (UB)
- [ ] 1.10 const / 引用 / move

> 自测文档已全部就绪，详见 `01_cpp_vs_python/`。
> 先自测，说不上来的提出来，我讲。
> 全部自测通过 = 勾上 ✅

### Stage 2 细化

- [ ] 2.1 `new`/`delete` 与 `malloc`/`free` 区别
- [ ] 2.2 RAII 手写实现
- [ ] 2.3 移动语义与完美转发
- [ ] 2.4 智能指针实战

## 📖 参考

已完成的老内容（旧学习路径）放到了 `_reference/`，你回头看用。

- `_reference/01_modern_cpp_basics/`
- `_reference/02_concurrency/`
- `_reference/03_system_programming/`
- `_reference/04_projects/`
- `_reference/multithreading/`
