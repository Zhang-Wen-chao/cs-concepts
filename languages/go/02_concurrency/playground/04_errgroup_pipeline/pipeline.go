package errgrouppipeline

import (
	"context"

	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
)

// ProcessBatch 使用 errgroup + 信号量并发处理输入，并保持结果顺序。
func ProcessBatch(ctx context.Context, inputs []string, limit int, fn func(context.Context, string) (string, error)) ([]string, error) {
	if fn == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = len(inputs)
		if limit == 0 {
			limit = 1
		}
	}

	g, ctx := errgroup.WithContext(ctx)
	sem := semaphore.NewWeighted(int64(limit))
	results := make([]string, len(inputs))

	for i, in := range inputs {
		i, in := i, in
		if err := sem.Acquire(ctx, 1); err != nil {
			return nil, err
		}
		g.Go(func() error {
			defer sem.Release(1)
			val, err := fn(ctx, in)
			if err != nil {
				return err
			}
			results[i] = val
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	return results, nil
}
