package crawler

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// Result captures a single fetch attempt summary.
type Result struct {
	URL       string
	Status    int
	Duration  time.Duration
	FetchedAt time.Time
	Err       error
}

// Fetcher defines the minimal contract for HTTP or fake fetchers.
type Fetcher interface {
	Fetch(ctx context.Context, url string) (Result, error)
}

// Run wires一个受限并发 worker pool，逐个抓取 URL 并返回结果。
func Run(ctx context.Context, urls []string, fetch Fetcher, maxWorkers int) ([]Result, error) {
	if fetch == nil {
		return nil, fmt.Errorf("fetcher is required")
	}
	if maxWorkers <= 0 {
		maxWorkers = runtime.NumCPU()
	}
	jobs := make(chan string)
	resultsCh := make(chan Result)

	var wg sync.WaitGroup
	worker := func() {
		defer wg.Done()
		for url := range jobs {
			res, err := fetch.Fetch(ctx, url)
			if err != nil {
				res = Result{URL: url, Err: err}
			}
			select {
			case <-ctx.Done():
				return
			case resultsCh <- res:
			}
		}
	}

	wg.Add(maxWorkers)
	for i := 0; i < maxWorkers; i++ {
		go worker()
	}

	go func() {
		defer close(jobs)
		for _, url := range urls {
			select {
			case <-ctx.Done():
				return
			case jobs <- url:
			}
		}
	}()

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	var results []Result
	for res := range resultsCh {
		results = append(results, res)
	}

	return results, ctx.Err()
}
