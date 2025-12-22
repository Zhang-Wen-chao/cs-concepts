package contextguard

import (
	"context"
	"time"
)

// RunWithTimeout 在独立 goroutine 中执行 fn，并在超时或 fn 完成时返回。
func RunWithTimeout(parent context.Context, timeout time.Duration, fn func(context.Context) error) error {
	if fn == nil {
		return nil
	}
	if timeout <= 0 {
		return fn(parent)
	}

	ctx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- fn(ctx)
	}()

	select {
	case <-ctx.Done():
		// 如果 fn 先返回会在 default case 捕获；否则返回 context 错误。
		select {
		case err := <-done:
			if err != nil {
				return err
			}
			return ctx.Err()
		default:
			return ctx.Err()
		}
	case err := <-done:
		return err
	}
}
