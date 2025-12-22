# syntax=docker/dockerfile:1
FROM golang:1.22 AS build
WORKDIR /src
COPY . .
RUN go build -o /out/todo-api ./01_todo_api/cmd/server

FROM gcr.io/distroless/base-debian12
COPY --from=build /out/todo-api /todo-api
EXPOSE 8080
ENTRYPOINT ["/todo-api"]
