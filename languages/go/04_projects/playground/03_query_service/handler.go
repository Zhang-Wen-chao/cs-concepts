package queryservice

import (
	"encoding/json"
	"net/http"
)

// Store 定义查询接口，Stage 4 中可由 CLI / API 共享。
type Store interface {
	List() ([]map[string]any, error)
}

// Handler 返回一个简易查询 HTTP handler。
func Handler(store Store) http.Handler {
	if store == nil {
		store = stubStore{}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		items, err := store.List()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"items": items})
	})
}

type stubStore struct{}

func (stubStore) List() ([]map[string]any, error) { return nil, nil }
