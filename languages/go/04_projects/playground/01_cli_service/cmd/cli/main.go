package main

import (
	"context"
	"flag"
	"log"

	"github.com/aaron/cs-concepts/go-projects/01_cli_service/internal/bridge"
)

func main() {
	var query string
	flag.StringVar(&query, "query", "", "query string sent to the analysis API")
	flag.Parse()

	client := bridge.NewClient("http://localhost:8080")
	if err := client.Run(context.Background(), query); err != nil {
		log.Fatalf("cli command failed: %v", err)
	}
}
