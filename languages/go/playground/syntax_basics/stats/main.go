package main

import (
	"fmt"
)

func main() {
	nums := []int{-3, -1, 4, 10}

	if report, err := describeNumbers(nums); err != nil {
		fmt.Println("describeNumbers error:", err)
	} else {
		fmt.Printf("count=%d sum=%d avg=%.2f class=%s\n", report.Count, report.Sum, report.Avg, report.Classification)
	}
}
