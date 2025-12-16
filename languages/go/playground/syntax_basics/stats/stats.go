package main

import "fmt"

// Report summarizes a collection of integers.
type Report struct {
	Count          int
	Sum            int
	Avg            float64
	Classification string
}

// describeNumbers calculates total, average, and a qualitative label.
func describeNumbers(nums []int) (Report, error) {
	if len(nums) == 0 {
		return Report{}, fmt.Errorf("empty input")
	}

	sum := 0
	for _, n := range nums {
		sum += n
	}

	avg := float64(sum) / float64(len(nums))
	class := classify(sum)

	return Report{Count: len(nums), Sum: sum, Avg: avg, Classification: class}, nil
}

func classify(sum int) string {
	switch {
	case sum < 0:
		return "negative"
	case sum == 0:
		return "zero"
	default:
		return "positive"
	}
}
