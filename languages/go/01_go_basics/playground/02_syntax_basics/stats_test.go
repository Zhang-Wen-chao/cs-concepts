package main

import "testing"

// TestDescribeNumbers 覆盖三种输入：正和、零和、空切片。
// t *testing.T 是测试上下文，提供 Fatal/Fatalf 等断言方法。
func TestDescribeNumbers(t *testing.T) {
	t.Parallel() // 允许此测试与同文件其他测试并行运行

	// cases 切片使用匿名 struct 描述输入/预期。字段解释：
	// - name: 子测试名称
	// - input: 函数输入的整数切片
	// - want: 期望的 Report
	// - wantErr: 是否预期返回 error
	cases := []struct {
		name    string
		input   []int
		want    Report
		wantErr bool
	}{
		{
			name:  "positive",
			input: []int{2, 3, 5},
			want:  Report{Count: 3, Sum: 10, Avg: 10.0 / 3.0, Classification: "positive"},
		},
		{
			name:  "zero",
			input: []int{-1, 0, 1},
			want:  Report{Count: 3, Sum: 0, Avg: 0, Classification: "zero"},
		},
		{
			name:    "empty",
			input:   nil, // nil 切片表示没有元素
			wantErr: true,
		},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()

			got, err := describeNumbers(c.input)
			if c.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// 逐字段比较：Count/Sum/Classification 都是整数或字符串，可直接 ==。
			if got.Count != c.want.Count || got.Sum != c.want.Sum || got.Classification != c.want.Classification {
				t.Fatalf("report mismatch: got %+v want %+v", got, c.want)
			}

			// Avg 是浮点，使用误差阈值比较。
			if diff := got.Avg - c.want.Avg; diff > 0.0001 || diff < -0.0001 {
				t.Fatalf("avg mismatch: got %f want %f", got.Avg, c.want.Avg)
			}
		})
	}
}

// TestClassify 单独测试 switch 分支，便于定位错误。
func TestClassify(t *testing.T) {
	t.Parallel()

	tests := []struct {
		sum  int
		want string
	}{
		{-1, "negative"},
		{0, "zero"},
		{5, "positive"},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.want, func(t *testing.T) {
			t.Parallel()

			if got := classify(tt.sum); got != tt.want {
				t.Fatalf("classify(%d)=%s want %s", tt.sum, got, tt.want)
			}
		})
	}
}
