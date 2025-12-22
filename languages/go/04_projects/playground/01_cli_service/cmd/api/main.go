package main

import (
	"log"
	"net/http"

	"github.com/aaron/cs-concepts/go-projects/01_cli_service/internal/bridge"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:    ":8080",
		Handler: bridge.NewRouter(mux),
	}

	log.Fatal(srv.ListenAndServe())
}
