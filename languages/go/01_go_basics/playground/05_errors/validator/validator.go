package validator

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

// ValidateUser 返回包裹后的错误，演示 errors.Join 与 fmt.Errorf("%w").
func ValidateUser(u User) error {
	var errs []error

	if strings.TrimSpace(u.Name) == "" {
		errs = append(errs, ErrEmptyName)
	}
	if u.Age < 0 || u.Age > 120 {
		errs = append(errs, fmt.Errorf("age %d: %w", u.Age, ErrInvalidAge))
	}
	if !strings.Contains(u.Email, "@") {
		errs = append(errs, fmt.Errorf("email %q: %w", u.Email, ErrInvalidEmail))
	}

	if len(errs) == 0 {
		return nil
	}

	if len(errs) == 1 {
		return errs[0]
	}

	return errors.Join(errs...)
}
