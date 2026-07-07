# 系统设计

## 架构

```
┌─────────────────────────────────────────────────┐
│                    log_analyzer                    │
│                                                   │
│  ┌──────┐  ┌──────┐  ┌────────┐  ┌───────────┐  │
│  │ CLI  │  │ HTTP │  │ Parser │  │ Aggregator │  │
│  │ Mode │  │ Mode │  │        │  │            │  │
│  └──┬───┘  └──┬───┘  └───┬────┘  └─────┬─────┘  │
│     │         │          │             │         │
│     └─────────┴──────────┴─────────────┘         │
│                         │                         │
│                   ┌─────▼──────┐                  │
│                   │   Storage   │                  │
│                   │ (in-memory) │                  │
│                   └────────────┘                  │
└─────────────────────────────────────────────────┘
```

## 数据流

### CLI 模式

```
file/stdin → LineReader → Parser → Filter → Aggregator → Formatter → stdout/file
```

### API 模式

```
HTTP POST /v1/ingest → Parser → Aggregator → InMemoryStore
HTTP GET  /v1/query  → Store   → QueryHandler → JSON Response
```

## 核心模块

| 模块 | 职责 |
|------|------|
| `cli.py` | Click/argparse 入口，协调管线 |
| `server.py` | FastAPI/Flask 应用，API 路由 |
| `parser.py` | 日志解析器（支持多格式） |
| `aggregator.py` | 时间窗口统计（计数、延迟分位值） |
| `storage.py` | 内存存储，支持并发读写 |
| `models.py` | 数据模型 + 类型注解 |
| `config.py` | 配置管理（默认值 → env → 文件） |

## 并发模型

- CLI 模式：同步，但解析/聚合阶段可用 `concurrent.futures` 多文件并行
- API 模式：asyncio + uvicorn，非阻塞处理
- 存储读写：`threading.Lock` 或 `asyncio.Lock` 保护共享状态
