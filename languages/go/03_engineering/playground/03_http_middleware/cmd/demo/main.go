package main

import (
	"fmt"
	"net/http"

	"github.com/aaron/cs-concepts/go-engineering/03_http_middleware/middleware"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/ping", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintln(w, "pong")
	})
	_ = middleware.Logging(mux)
	fmt.Println("middleware wired; integrate into server in Stage 3")
}
