package main

import "testing"

func TestDescribeNumbers(t *testing.T) {
	t.Parallel()

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
			input:   nil,
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

			if got.Count != c.want.Count || got.Sum != c.want.Sum || got.Classification != c.want.Classification {
				t.Fatalf("report mismatch: got %+v want %+v", got, c.want)
			}

			if diff := got.Avg - c.want.Avg; diff > 0.0001 || diff < -0.0001 {
				t.Fatalf("avg mismatch: got %f want %f", got.Avg, c.want.Avg)
			}
		})
	}
}

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
