package main

import (
	"context"
	"fmt"

	"github.com/aaron/cs-concepts/go-engineering/02_tooling_runner/runner"
)

func main() {
	ctx := context.Background()
	_ = runner.Run(ctx, nil, func(_ context.Context, args []string) error {
		fmt.Println("would run:", args)
		return nil
	})
}
