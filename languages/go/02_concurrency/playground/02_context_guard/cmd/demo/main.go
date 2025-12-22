package main

import (
	"context"
	"fmt"
	"time"

	"github.com/aaron/cs-concepts/go-concurrency/02_context_guard"
)

func main() {
	err := contextguard.RunWithTimeout(context.Background(), 50*time.Millisecond, func(ctx context.Context) error {
		select {
		case <-time.After(100 * time.Millisecond):
			fmt.Println("work completed")
			return nil
		case <-ctx.Done():
			fmt.Println("context canceled:", ctx.Err())
			return ctx.Err()
		}
	})

	fmt.Println("result:", err)
}
