# 阶段 4 · 综合项目

> 目标：完成 “CLI + Service” 组合项目（日志分析/可观测性工具示例），囊括需求设计、数据流、集成测试、容器化部署与复盘。

## 学习闭环
1. **需求定义**：`notes/01_project_brief.md` 写清角色、核心场景、成功指标。
2. **系统设计**：在 `notes/02_system_design.md` 绘制架构/数据流/接口协议，确定并发模型。
3. **实现**：`playground/01_cli_service`（单一二进制，`--mode=api|cli` 切换）+ `02_ingest_pipeline` + `03_query_service`，与 Stage 2/3 的组件复用。
4. **验证**：端到端测试（`httptest` + CLI golden files）+ 性能压测 + docker-compose。
5. **复盘**：记录 learning log、性能瓶颈、跨语言互操作（如 gRPC/HTTP）。

## Checklist
- [ ] CLI 支持 flag + config file + 环境变量，并能与 API 对话（HTTP/JSON）。
- [ ] API 具备 `/healthz`、`/metrics`、业务 endpoints，内建 observability middleware。
- [ ] 统一的 `01_cli_service` 内桥接代码：抽象协议/DTO，便于 C++/Python 组件对接。
- [ ] 集成测试：API handler 用 `httptest` 自定义 server，CLI 用 golden file + fake API。
- [ ] 性能评估：记录并行度、最大吞吐、资源占用，必要时使用 `pprof`/`trace`。
- [ ] 部署：Docker Compose（API + CLI job + 可选数据库），或 Makefile 一键跑。
- [ ] 文档：README（项目概述）、架构图（可用 Mermaid）、操作手册、复盘纪要。

## Playground 模块
- `01_cli_service` —— CLI + API 主项目骨架（`--mode=api|cli`）。
- `02_ingest_pipeline/...` —— 日志解析与路径聚合示例。
- `03_query_service/...` —— 基于接口解耦的查询 HTTP handler，配合集成测试使用。

## 验收方式
```bash
cd languages/go/04_projects/playground
go test ./... -run Test -race
go run 01_cli_service --mode=api &
go run 01_cli_service --mode=cli --query \"error\" | tee out.log
docker build -t go-cli-service .
```
- CLI 能够列出 API 返回的统计信息，并在 API 停机时给出友好错误。
- 终端或 README 中附带集成测试与部署截图/日志。

## 复盘要点
- 项目分层是否清晰（domain vs transport vs adapter）。
- 并发与容错设计是否借鉴了 Stage 2/3 的经验。
- 是否存在可以抽象成公共库的部分，为未来语言栈复用做铺垫。
