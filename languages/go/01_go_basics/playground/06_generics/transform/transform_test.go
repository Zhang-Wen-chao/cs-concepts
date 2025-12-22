package transform

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
)

func TestMapSlice(t *testing.T) {
	t.Parallel()

	got := MapSlice([]int{1, 2, 3}, func(i int) string {
		return fmt.Sprintf("n=%d", i)
	})

	want := []string{"n=1", "n=2", "n=3"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("MapSlice mismatch: %v", got)
	}
}

func TestFilter(t *testing.T) {
	t.Parallel()

	got := Filter([]int{1, 2, 3, 4}, func(i int) bool { return i%2 == 0 })
	want := []int{2, 4}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Filter mismatch: %v", got)
	}
}

func TestKeys(t *testing.T) {
	t.Parallel()

	keys := Keys(map[string]int{"go": 1, "cpp": 2})
	sort.Strings(keys)
	want := []string{"cpp", "go"}
	if !reflect.DeepEqual(keys, want) {
		t.Fatalf("Keys mismatch: %v", keys)
	}
}
