package crawler

import (
	"context"
	"net/http"
	"time"
)

// HTTPFetcher 使用 net/http 抓取 URL 并记录状态/耗时。
type HTTPFetcher struct {
	Client *http.Client
}

// Fetch implements the Fetcher interface.
func (h *HTTPFetcher) Fetch(ctx context.Context, url string) (Result, error) {
	client := h.Client
	if client == nil {
		client = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Result{URL: url}, err
	}

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return Result{URL: url, Duration: time.Since(start)}, err
	}
	defer resp.Body.Close()

	return Result{
		URL:       url,
		Status:    resp.StatusCode,
		Duration:  time.Since(start),
		FetchedAt: time.Now(),
	}, nil
}
