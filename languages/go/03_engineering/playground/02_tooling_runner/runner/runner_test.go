package runner

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestRunUsesDefaultSteps(t *testing.T) {
	t.Parallel()

	var seen [][]string
	err := Run(context.Background(), nil, func(_ context.Context, args []string) error {
		copyArgs := append([]string(nil), args...)
		seen = append(seen, copyArgs)
		return nil
	})
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	want := [][]string{{"go", "fmt", "./..."}, {"go", "vet", "./..."}, {"go", "test", "./..."}}
	if !reflect.DeepEqual(seen, want) {
		t.Fatalf("unexpected commands: %v", seen)
	}
}

func TestRunStopsOnError(t *testing.T) {
	t.Parallel()

	fakeErr := errors.New("boom")
	err := Run(context.Background(), DefaultSteps(), func(_ context.Context, args []string) error {
		if args[1] == "vet" {
			return fakeErr
		}
		return nil
	})
	if !errors.Is(err, fakeErr) {
		t.Fatalf("expected error propagation, got %v", err)
	}
}
