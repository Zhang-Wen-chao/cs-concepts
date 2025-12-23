package validator // 错误处理套路：哨兵错误 + %w + errors.Join

/*
Usage:

	cd languages/go/01_go_basics/playground/05_errors
	go fmt ./...
	go test ./...
	go test . -run TestValidateUserJoin
*/

import (
	"errors"
	"fmt"
	"strings"
)

// 自定义错误，用于 errors.Is 匹配。
var (
	ErrEmptyName    = errors.New("empty name")
	ErrInvalidAge   = errors.New("invalid age")
	ErrInvalidEmail = errors.New("invalid email")
)

// User 示例模型。
type User struct {
	Name  string
	Age   int
	Email string
}

// ValidateUser 返回包裹后的错误，演示 errors.Join 与 fmt.Errorf("%w")。
// 签名：func ValidateUser(u User) error —— 参数是单个 User 值，返回 error（nil 表示通过校验）。
func ValidateUser(u User) error {
	var errs []error

	if strings.TrimSpace(u.Name) == "" {
		// 直接附加哨兵错误，方便外层判断。
		errs = append(errs, ErrEmptyName)
	}
	if u.Age < 0 || u.Age > 120 {
		// fmt.Errorf("%w", ErrInvalidAge) 会把上下文与哨兵链接在一起。
		errs = append(errs, fmt.Errorf("age %d: %w", u.Age, ErrInvalidAge))
	}
	if !strings.Contains(u.Email, "@") {
		errs = append(errs, fmt.Errorf("email %q: %w", u.Email, ErrInvalidEmail))
	}

	if len(errs) == 0 {
		return nil
	}

	if len(errs) == 1 {
		// 只有一个错误时没必要 Join，直接返回即可，减少包裹层级。
		return errs[0]
	}

	// errors.Join 会把多个错误聚合在一起，仍可被 errors.Is/As 匹配。
	return errors.Join(errs...)
}
