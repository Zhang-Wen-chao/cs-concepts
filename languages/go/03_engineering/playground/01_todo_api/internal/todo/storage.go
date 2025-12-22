package todo

import "context"

// Repository defines persistence operations so we can swap memory/SQL later.
type Repository interface {
	List(ctx context.Context, filter ListFilter) ([]Item, error)
	Save(ctx context.Context, item Item) error
}

// Item captures a Todo entity.
type Item struct {
	ID     string
	Title  string
	Status string
}

// ListFilter controls List queries.
type ListFilter struct {
	Status string
	Limit  int
	Offset int
}

// MemoryRepo provides a starting point for tests.
type MemoryRepo struct {
	items []Item
}

// NewMemoryRepo returns an empty in-memory repository.
func NewMemoryRepo() *MemoryRepo {
	return &MemoryRepo{}
}

// List implements Repository.
func (r *MemoryRepo) List(ctx context.Context, filter ListFilter) ([]Item, error) {
	_ = ctx
	_ = filter
	// TODO: implement filtering/pagination.
	return append([]Item(nil), r.items...), nil
}

// Save adds/updates an item in memory.
func (r *MemoryRepo) Save(ctx context.Context, item Item) error {
	_ = ctx
	r.items = append(r.items, item)
	return nil
}
