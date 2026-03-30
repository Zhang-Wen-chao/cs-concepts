package contextguard

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRunWithTimeoutHitsDeadline(t *testing.T) {
	t.Parallel()

	err := RunWithTimeout(context.Background(), 10*time.Millisecond, func(ctx context.Context) error {
		select {
		case <-time.After(30 * time.Millisecond):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline exceeded, got %v", err)
	}
}

func TestRunWithTimeoutReturnsFuncError(t *testing.T) {
	t.Parallel()

	want := errors.New("boom")
	err := RunWithTimeout(context.Background(), time.Second, func(context.Context) error {
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("expected %v, got %v", want, err)
	}
}

func TestRunWithTimeoutNilFunc(t *testing.T) {
	t.Parallel()

	if err := RunWithTimeout(context.Background(), time.Second, nil); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
}
