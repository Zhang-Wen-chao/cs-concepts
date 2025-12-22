package main

import (
	"fmt"

	"github.com/aaron/cs-concepts/go-playground/04_interfaces/shapes"
)

func main() {
	fmt.Println(shapes.Describe(shapes.Rectangle{Width: 3, Height: 4}))
}
