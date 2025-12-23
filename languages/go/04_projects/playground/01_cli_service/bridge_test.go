package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClientRun(t *testing.T) {
	t.Parallel()

	client := NewClient("http://example.org")
	if err := client.Run(context.Background(), "demo"); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
}

func TestNewRouterPassthrough(t *testing.T) {
	t.Parallel()

	called := false
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusAccepted)
	})

	router := NewRouter(mux)
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rr := httptest.NewRecorder()

	router.ServeHTTP(rr, req)

	if !called {
		t.Fatalf("expected underlying handler to be invoked")
	}

	if rr.Code != http.StatusAccepted {
		t.Fatalf("unexpected status code: %d", rr.Code)
	}
}
