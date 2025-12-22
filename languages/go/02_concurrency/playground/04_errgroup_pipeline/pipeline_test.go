package errgrouppipeline

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestProcessBatchKeepsOrder(t *testing.T) {
	t.Parallel()

	inputs := []string{"a", "b", "c"}
	ctx := context.Background()
	var maxParallel int64
	var current int64

	results, err := ProcessBatch(ctx, inputs, 2, func(ctx context.Context, s string) (string, error) {
		cur := atomic.AddInt64(&current, 1)
		for {
			old := atomic.LoadInt64(&maxParallel)
			if cur <= old {
				break
			}
			if atomic.CompareAndSwapInt64(&maxParallel, old, cur) {
				break
			}
		}
		defer atomic.AddInt64(&current, -1)
		time.Sleep(5 * time.Millisecond)
		return s + "!", nil
	})

	if err != nil {
		t.Fatalf("ProcessBatch error: %v", err)
	}
	if got, want := results, []string{"a!", "b!", "c!"}; !equal(got, want) {
		t.Fatalf("unexpected order: %v", got)
	}
	if maxParallel > 2 {
		t.Fatalf("limit exceeded: %d", maxParallel)
	}
}

func TestProcessBatchReturnsError(t *testing.T) {
	t.Parallel()

	want := errors.New("boom")
	_, err := ProcessBatch(context.Background(), []string{"x"}, 1, func(context.Context, string) (string, error) {
		return "", want
	})
	if !errors.Is(err, want) {
		t.Fatalf("expected %v, got %v", want, err)
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
