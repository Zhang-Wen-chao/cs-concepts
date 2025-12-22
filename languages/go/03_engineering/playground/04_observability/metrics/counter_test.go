package metrics

import "testing"

func TestCounterRegistry(t *testing.T) {
	t.Parallel()

	reg := NewCounterRegistry()
	reg.Add("requests", 2)
	reg.Add("requests", 3)
	reg.Add("errors", 1)

	snap := reg.Snapshot()
	if snap["requests"] != 5 {
		t.Fatalf("expected 5 requests, got %d", snap["requests"])
	}
	if snap["errors"] != 1 {
		t.Fatalf("expected 1 error, got %d", snap["errors"])
	}
}
