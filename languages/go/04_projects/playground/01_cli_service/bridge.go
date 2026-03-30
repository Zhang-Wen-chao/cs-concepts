package main

import (
	"context"
	"fmt"
	"net/http"
)

// Client represents the CLI <-> API contract.
type Client struct {
	baseURL string
	http    *http.Client
}

// NewClient wires default HTTP client.
func NewClient(baseURL string) *Client {
	return &Client{baseURL: baseURL, http: &http.Client{}}
}

// Run sends the query to the API (placeholder implementation).
func (c *Client) Run(ctx context.Context, query string) error {
	_ = ctx
	fmt.Printf("sending query %q to %s\n", query, c.baseURL)
	return nil
}

// NewRouter decorates the provided mux with placeholder middleware.
func NewRouter(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// TODO: tracing/logging/metrics hooks.
		next.ServeHTTP(w, r)
	})
}
