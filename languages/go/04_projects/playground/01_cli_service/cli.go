package main

import (
	"context"
)

func runCLI(baseURL, query string) error {
	client := NewClient(baseURL)
	return client.Run(context.Background(), query)
}
