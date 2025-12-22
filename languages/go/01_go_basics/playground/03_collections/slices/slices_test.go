package slices

import "testing"

func TestChunk(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		nums    []int
		size    int
		wantLen int
		wantErr bool
	}{
		{name: "even", nums: []int{1, 2, 3, 4}, size: 2, wantLen: 2},
		{name: "leftover", nums: []int{1, 2, 3, 4, 5}, size: 2, wantLen: 3},
		{name: "empty", nums: nil, size: 3, wantLen: 0},
		{name: "invalid", nums: []int{1}, size: 0, wantErr: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := Chunk(tc.nums, tc.size)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != tc.wantLen {
				t.Fatalf("chunk length mismatch: got %d want %d", len(got), tc.wantLen)
			}
			for _, chunk := range got {
				if len(chunk) > tc.size && len(tc.nums) > 0 {
					t.Fatalf("chunk size exceeded: %v", chunk)
				}
			}
			if len(tc.nums) > 0 {
				tc.nums[0] = 999
				if got[0][0] == 999 {
					t.Fatalf("chunks should be copy, got aliasing")
				}
			}
		})
	}
}

func TestMergeCounters(t *testing.T) {
	t.Parallel()

	base := map[string]int{"go": 1}
	delta := map[string]int{"go": 2, "cpp": 3}

	merged := MergeCounters(base, delta)
	if merged["go"] != 3 || merged["cpp"] != 3 {
		t.Fatalf("unexpected merged map: %#v", merged)
	}

	if base["cpp"] != 3 {
		t.Fatalf("expected base map mutated (reference semantics)")
	}
}
