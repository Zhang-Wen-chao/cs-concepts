package main

import (
	"fmt"

	"github.com/aaron/cs-concepts/go-playground/03_collections/slices"
)

func main() {
	chunks, _ := slices.Chunk([]int{1, 2, 3, 4, 5}, 2)
	fmt.Println("chunks:", chunks)
}
