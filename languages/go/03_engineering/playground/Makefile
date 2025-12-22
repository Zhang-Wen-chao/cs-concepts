SHELL := /bin/bash

.PHONY: fmt lint test bench cover run docker

fmt:
	go fmt ./...

lint:
	go vet ./...

test:
	go test ./... -race

bench:
	go test ./... -run Test -bench . -benchmem

cover:
	go test ./... -coverprofile=coverage.out
	go tool cover -func=coverage.out

run:
	go run ./01_todo_api/cmd/server

docker:
	docker build -t todo-api:dev .
