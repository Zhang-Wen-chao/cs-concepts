# 05 · 部署准备

> 资料：Go 官方 Dockerfile 示例、Google Cloud Buildpacks、AWS CDK Go 指南。

## Build Pipeline
1. `make lint` → fmt/vet/lint
2. `make test` → `go test ./... -race`
3. `make build` → `GOOS=linux GOARCH=amd64 go build -o bin/app ./01_todo_api`
4. `make docker` → `docker build -t todo-api:dev .`

## Docker 多阶段示例
```dockerfile
FROM golang:1.22 AS build
WORKDIR /src
COPY . .
RUN go build -o /out/server ./01_todo_api

FROM gcr.io/distroless/base-debian12
COPY --from=build /out/server /server
ENTRYPOINT [\"/server\"]
```
- 将配置文件/静态资源通过 `COPY` 加入最终镜像。
- 使用 distroless/alpine 缩小镜像体积。

## 发布与配置
- 通过环境变量注入端口、数据库 URL、日志级别。
- 支持健康检查：Kubernetes `readinessProbe`/`livenessProbe`。
- 记录 `git commit` 与 build time：`-ldflags \"-X main.version=$(git rev-parse --short HEAD)\"`.

## Checklist
- [ ] 本地 `make build` 生成静态二进制，可运行在容器内。
- [ ] Dockerfile 支持多阶段，最终镜像 < 50MB。
- [ ] README 写明部署步骤 + 环境变量说明。
