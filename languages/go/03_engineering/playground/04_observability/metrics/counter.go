package metrics

import (
	"expvar"
	"sync"
)

// CounterRegistry 暴露 expvar 计数器。
type CounterRegistry struct {
	mu   sync.Mutex
	vars map[string]*expvar.Int
}

// NewCounterRegistry 创建注册表。
func NewCounterRegistry() *CounterRegistry {
	return &CounterRegistry{vars: make(map[string]*expvar.Int)}
}

// Add 增加指定计数器。
func (r *CounterRegistry) Add(name string, delta int64) {
	if name == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	v := r.vars[name]
	if v == nil {
		v = expvar.NewInt(name)
		r.vars[name] = v
	}
	v.Add(delta)
}

// Snapshot 返回当前计数器值副本。
func (r *CounterRegistry) Snapshot() map[string]int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[string]int64, len(r.vars))
	for k, v := range r.vars {
		out[k] = v.Value()
	}
	return out
}
