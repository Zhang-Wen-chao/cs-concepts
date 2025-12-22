# 01 · CLI + Service 组合项目概述

> 案例：日志分析 CLI + HTTP 查询服务。CLI 读取本地/远程日志，聚合指标后推送到服务；服务提供搜索/聚合 API，并导出健康检查与指标。

## 角色与场景
- **CLI 使用者**：SRE/开发者，需要快速分析日志并将结果同步给 Web 控制台。
- **服务消费者**：Web UI、自动化脚本，通过 REST/JSON 查询聚合结果。

## 成功指标
| 指标 | 目标 |
| --- | --- |
| CLI 每秒处理行数 | >= 50k lines/s（本地） |
| API 99% 延迟 | < 150ms |
| 数据一致性 | CLI 成功推送的数据可在 1s 内被 API 查询到 |

## 交付物
- 双可执行文件：`01_cli_service/cmd/cli`、`01_cli_service/cmd/api`
- 公共 `01_cli_service/internal/bridge` 包：统一 DTO/HTTP 客户端
- 运行手册 + API 文档（可用 OpenAPI 或简单 Markdown）
- docker-compose：一键启动 API + 可选数据库 + fake log producer

## 时间分配
1. 需求 & 设计（0.5 天）
2. 数据管道（1 天）：文件/STDIN → parser → aggregator → HTTP push
3. API + 存储（1 天）：内存/SQLite，暴露查询接口
4. 观测 + 部署（0.5 天）：日志、metrics、Docker、README
5. 复盘（0.5 天）：性能、可维护性、下一步计划
