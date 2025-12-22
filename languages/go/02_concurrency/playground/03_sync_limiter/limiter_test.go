package synclimiter

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestLimiterEnforcesLimit(t *testing.T) {
	t.Parallel()

	limiter := NewLimiter(2)
	ctx := context.Background()

	var inFlight int64
	var max int64
	var wg sync.WaitGroup

	worker := func() error {
		return limiter.Do(ctx, func(context.Context) error {
			cur := atomic.AddInt64(&inFlight, 1)
			for {
				old := atomic.LoadInt64(&max)
				if cur <= old {
					break
				}
				if atomic.CompareAndSwapInt64(&max, old, cur) {
					break
				}
			}
			time.Sleep(10 * time.Millisecond)
			atomic.AddInt64(&inFlight, -1)
			return nil
		})
	}

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := worker(); err != nil {
				t.Errorf("worker error: %v", err)
			}
		}()
	}

	wg.Wait()

	if max > 2 {
		t.Fatalf("expected max concurrency <=2, got %d", max)
	}
}

func TestLimiterRespectsContextCancellation(t *testing.T) {
	t.Parallel()

	limiter := NewLimiter(1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := limiter.Do(ctx, func(context.Context) error { return nil })
	if err == nil {
		t.Fatalf("expected context error")
	}
}
