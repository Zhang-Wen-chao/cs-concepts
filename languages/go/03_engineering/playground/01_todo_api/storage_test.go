package main

import (
	"context"
	"testing"
)

func TestMemoryRepoSaveAndList(t *testing.T) {
	t.Parallel()

	repo := NewMemoryRepo()
	ctx := context.Background()

	if err := repo.Save(ctx, Item{ID: "1", Title: "demo", Status: "pending"}); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	items, err := repo.List(ctx, ListFilter{})
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}

	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
}
