package queryservice

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

type fakeStore struct {
	items []map[string]any
	err   error
}

func (f fakeStore) List() ([]map[string]any, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.items, nil
}

func TestHandlerReturnsJSON(t *testing.T) {
	t.Parallel()

	store := fakeStore{items: []map[string]any{{"path": "/login"}}}
	req := httptest.NewRequest(http.MethodGet, "/query", nil)
	rr := httptest.NewRecorder()

	Handler(store).ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", rr.Code)
	}
	if got := rr.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("unexpected content type: %s", got)
	}
}

func TestHandlerHandlesError(t *testing.T) {
	t.Parallel()

	store := fakeStore{err: errors.New("db down")}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/query", nil)

	Handler(store).ServeHTTP(rr, req)
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", rr.Code)
	}
}
