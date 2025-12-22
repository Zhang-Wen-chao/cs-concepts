package main

import (
	"log"

	"github.com/aaron/cs-concepts/go-engineering/01_todo_api/internal/todo"
)

func main() {
	srv := todo.NewServer()
	log.Fatal(srv.Start())
}
