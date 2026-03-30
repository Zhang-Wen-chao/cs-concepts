package slices

import "testing"

// 运行方式：`go test ./...` 或 `go test . -run TestChunk`，默认会并行执行每个子测试。

func TestChunk(t *testing.T) {
	t.Parallel()

	// table-driven 测试模板：用匿名 struct 描述输入/预期，让新增场景时只需 append。
	cases := []struct {
		name    string // 子测试名称
		nums    []int  // 输入切片
		size    int    // 每段的最大长度
		wantLen int    // 期望返回的分段数量
		wantErr bool   // 是否预期错误
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
					// 如果修改输入导致输出同时变化，说明漏掉了 copy，立刻报错。
					t.Fatalf("chunks should be copy, got aliasing")
				}
			}
		})
	}
}

func TestMergeCounters(t *testing.T) {
	t.Parallel()

	// map 是引用语义：两个变量指向同一底层结构，因此只需要操作 base。
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
