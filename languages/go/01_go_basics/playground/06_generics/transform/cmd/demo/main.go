package main

import (
	"fmt"

	"github.com/aaron/cs-concepts/go-playground/06_generics/transform"
)

func main() {
	result := transform.Filter(transform.MapSlice([]int{1, 2, 3}, func(i int) int { return i * 10 }), func(i int) bool { return i >= 20 })
	fmt.Println(result)
}
