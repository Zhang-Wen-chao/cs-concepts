package runner

import (
	"context"
	"fmt"
)

// Step 表示一个命令步骤，如 ["go", "fmt", "./..."].
type Step struct {
	Name string
	Args []string
}

// ExecFunc 允许在测试中注入不同实现。
type ExecFunc func(ctx context.Context, args []string) error

// DefaultSteps 返回基础工具链顺序。
func DefaultSteps() []Step {
	return []Step{
		{Name: "fmt", Args: []string{"go", "fmt", "./..."}},
		{Name: "vet", Args: []string{"go", "vet", "./..."}},
		{Name: "test", Args: []string{"go", "test", "./..."}},
	}
}

// Run 按顺序执行步骤，遇到错误立即返回。
func Run(ctx context.Context, steps []Step, execFn ExecFunc) error {
	if execFn == nil {
		return fmt.Errorf("execFn is nil")
	}
	if len(steps) == 0 {
		steps = DefaultSteps()
	}
	for _, step := range steps {
		args := append([]string(nil), step.Args...)
		if err := execFn(ctx, args); err != nil {
			return fmt.Errorf("%s failed: %w", step.Name, err)
		}
	}
	return nil
}
