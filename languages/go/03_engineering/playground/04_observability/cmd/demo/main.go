package main

import (
	"fmt"
	"net/http"

	"github.com/aaron/cs-concepts/go-engineering/04_observability/metrics"
)

func main() {
	reg := metrics.NewCounterRegistry()
	reg.Add("requests", 1)
	fmt.Println("snapshot:", reg.Snapshot())

	http.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		for k, v := range reg.Snapshot() {
			fmt.Fprintf(w, "%s %d\n", k, v)
		}
	})
	fmt.Println("metrics handler registered; plug into real server in Stage 3")
}
