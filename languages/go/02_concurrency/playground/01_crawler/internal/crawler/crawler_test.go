package crawler

import (
	"context"
	"errors"
	"testing"
	"time"
)

type fakeFetcher struct {
	delay time.Duration
}

func (f fakeFetcher) Fetch(ctx context.Context, url string) (Result, error) {
	select {
	case <-ctx.Done():
		return Result{URL: url}, ctx.Err()
	case <-time.After(f.delay):
	}
	return Result{
		URL:       url,
		Status:    200,
		Duration:  f.delay,
		FetchedAt: time.Now(),
	}, nil
}

func TestRunProcessesAllURLs(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	fetch := fakeFetcher{delay: 5 * time.Millisecond}
	urls := []string{"https://example.org/a", "https://example.org/b"}

	got, err := Run(ctx, urls, fetch, 2)
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	if len(got) != len(urls) {
		t.Fatalf("expected %d results, got %d", len(urls), len(got))
	}
	for _, res := range got {
		if res.Err != nil {
			t.Fatalf("unexpected error in result: %v", res.Err)
		}
		if res.Status != 200 {
			t.Fatalf("unexpected status %d", res.Status)
		}
	}
}

func TestRunHonorsCanceledContext(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := Run(ctx, []string{"https://example.org"}, fakeFetcher{}, 1)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}
