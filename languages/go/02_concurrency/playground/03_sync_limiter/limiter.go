package synclimiter

import (
	"context"
)

// Limiter 使用带缓冲 channel 控制并发度。
type Limiter struct {
	tokens chan struct{}
}

// NewLimiter 创建一个最大并发为 n 的 limiter；n<=0 时回退为 1。
func NewLimiter(n int) *Limiter {
	if n <= 0 {
		n = 1
	}
	return &Limiter{tokens: make(chan struct{}, n)}
}

// Do 尝试在上下文未取消的情况下执行 fn。
func (l *Limiter) Do(ctx context.Context, fn func(context.Context) error) error {
	if fn == nil {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case l.tokens <- struct{}{}:
	}
	defer func() { <-l.tokens }()
	return fn(ctx)
}

// Available 返回当前可用令牌数，便于测试观察。
func (l *Limiter) Available() int { return cap(l.tokens) - len(l.tokens) }
