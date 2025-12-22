package main

import (
	"context"
	"fmt"
	"sync"
	"time"

	synclimiter "github.com/aaron/cs-concepts/go-concurrency/03_sync_limiter"
)

func main() {
	limiter := synclimiter.NewLimiter(2)
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			limiter.Do(ctx, func(context.Context) error {
				fmt.Printf("worker %d running (available=%d)\n", id, limiter.Available())
				time.Sleep(50 * time.Millisecond)
				return nil
			})
		}(i)
	}
	wg.Wait()
}
