package main

import (
	"fmt"

	"github.com/aaron/cs-concepts/go-playground/05_errors/validator"
)

func main() {
	if err := validator.ValidateUser(validator.User{}); err != nil {
		fmt.Println("validation failed:", err)
	}
}
