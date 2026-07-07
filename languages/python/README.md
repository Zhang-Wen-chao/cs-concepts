# Python 学习路径

> 你不是 Python 新手。这里补的是 Python 深度知识 + 与 C++/Go 的对比。

## 📂 目录结构

| 阶段 | 目录 | 内容 |
|---|---|---|
| 01 基础但不基础 | `01_foundations/` | 思维模式、类型系统、函数深入、OOP、异常、import |
| 02 进阶能力 | `02_intermediate/` | 装饰器、上下文管理器、生成器、async、元类、descriptor、coroutine 底层 |
| 03 工程化 | `03_engineering/` | 模块与包、测试、类型注解、性能分析 |
| 04 项目实战 | `04_projects/` | 日志分析 CLI + API 服务（对标 Go 项目） |

每个阶段包含：
- `notes/` — 完整笔记
- `questions/` — 自测题
- `playground/` — 可运行代码 + pytest（仅部分阶段有）

一键运行所有测试：

```bash
cd languages/python
pip install -e 04_projects/01_log_analyzer
pytest
```

## 新增内容（本次补充）

| 内容 | 说明 |
|------|------|
| `02_intermediate/notes/06_descriptors.md` | Descriptor 协议：`__get__`/`__set__`、data vs non-data、property 本质 |
| `02_intermediate/notes/07_coroutine_deep.md` | Coroutine 底层：generator-based、yield from、await 本质、手写事件循环 |
| `01_foundations/playground/` | 类型系统、函数、OOP、异常 — 各模块含 pytest 测试 |
| `02_intermediate/playground/` | 装饰器、上下文管理器、生成器、async、元类、descriptor — 含 pytest + pytest-asyncio |
| `03_engineering/playground/` | 测试、类型注解、性能分析 — 含 pytest |
| `04_projects/` | **新阶段**：完整项目 "日志分析 CLI + API 服务"（可运行、有测试） |
| `pyproject.toml` | 统一测试配置 |

## ✅ 进度

- [x] 01_foundations — 6 篇笔记 + 1 套自测 + playground (4 模块，18 测试)
- [x] 02_intermediate — 7 篇笔记 + playground (6 模块，含 async)
- [x] 03_engineering — 4 篇笔记 + playground (3 模块)
- [x] 04_projects — 日志分析器 (parser/aggregator/server/cli + 14 测试)
