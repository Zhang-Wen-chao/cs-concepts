package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

func main() {
	urlsFile := flag.String("urls", "", "path to a file that lists URLs (one per line)")
	timeout := flag.Duration("timeout", 10*time.Second, "overall crawl timeout")
	maxWorkers := flag.Int("max-workers", 10, "maximum number of concurrent workers")
	flag.Parse()

	urls, err := loadURLs(*urlsFile, flag.Args())
	if err != nil {
		log.Fatalf("load urls: %v", err)
	}
	if len(urls) == 0 {
		log.Fatal("no url provided (use --urls file or pass urls as args)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	fetcher := &HTTPFetcher{}
	results, runErr := Run(ctx, urls, fetcher, *maxWorkers)
	if runErr != nil && runErr != context.Canceled && runErr != context.DeadlineExceeded {
		log.Fatalf("crawler failed: %v", runErr)
	}

	for _, res := range results {
		if res.Err != nil {
			fmt.Printf("FAIL\t%s\t%s\n", res.URL, res.Err)
			continue
		}
		fmt.Printf("OK\t%s\tstatus=%d\tlatency=%s\n", res.URL, res.Status, res.Duration)
	}
}

func loadURLs(path string, args []string) ([]string, error) {
	if path == "" && len(args) > 0 {
		return args, nil
	}
	if path == "" {
		return []string{"https://example.org"}, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var urls []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		urls = append(urls, line)
	}
	return urls, scanner.Err()
}
