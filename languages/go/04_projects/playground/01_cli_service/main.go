package main

import (
	"flag"
	"log"
)

func main() {
	mode := flag.String("mode", "api", "run mode: api or cli")
	addr := flag.String("addr", ":8080", "api listen address (api mode)")
	baseURL := flag.String("base-url", "http://localhost:8080", "API base URL (cli mode)")
	query := flag.String("query", "", "query string sent to the analysis API (cli mode)")
	flag.Parse()

	switch *mode {
	case "api":
		if err := runAPIServer(*addr); err != nil {
			log.Fatalf("api mode failed: %v", err)
		}
	case "cli":
		if err := runCLI(*baseURL, *query); err != nil {
			log.Fatalf("cli mode failed: %v", err)
		}
	default:
		log.Fatalf("unknown mode %q (use api or cli)", *mode)
	}
}
