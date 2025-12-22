package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	pipeline "github.com/aaron/cs-concepts/go-concurrency/04_errgroup_pipeline"
)

func main() {
	inputs := []string{"go", "cpp", "python"}
	results, err := pipeline.ProcessBatch(context.Background(), inputs, 2, func(ctx context.Context, s string) (string, error) {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
		return strings.ToUpper(s), nil
	})
	fmt.Println("results:", results, "err:", err)
}
