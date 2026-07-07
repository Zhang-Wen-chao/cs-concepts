# 项目一：日志分析 CLI + API 服务

> 与 Go 04_projects 相同命题，用 Python 实现，便于对比两语言的工程化差异。

## 需求

一个日志分析工具，同时提供 CLI 和 HTTP API 两种使用方式：

**CLI 模式**：从文件或 stdin 读取日志，支持过滤、聚合、输出到终端或文件。

```
python -m log_analyzer --mode=cli --input access.log --format=json
python -m log_analyzer --mode=cli --input access.log --filter="ERROR|FATAL"
```

**API 模式**：启动 HTTP 服务，接收日志上报、提供查询接口。

```
python -m log_analyzer --mode=api --port=8080
```

## 功能

1. **日志解析**：解析常见日志格式（Apache/Nginx combined、JSON、自定义）
2. **过滤**：按级别、时间范围、关键字过滤
3. **聚合**：按时间窗口（1m/5m/1h）统计请求量、错误率、P50/P90/P99 延迟
4. **CLI 输出**：表格、JSON、CSV
5. **API 端点**：`POST /v1/ingest` 上报日志、`GET /v1/query` 查询统计
6. **健康检查**：`GET /healthz`、`GET /metrics`

## 学习目标

- Python 包结构设计（src/ layout）
- Click/argparse CLI 框架
- FastAPI / Flask HTTP 服务
- asyncio 并发处理
- pytest + httpx 集成测试
- 类型注解 + mypy 检查
- Docker 容器化部署
