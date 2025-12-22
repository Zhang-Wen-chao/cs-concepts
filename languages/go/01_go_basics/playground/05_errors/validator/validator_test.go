package validator

import (
	"errors"
	"testing"
)

func TestValidateUser(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		user    User
		wantErr error
	}{
		{name: "ok", user: User{Name: "Go", Age: 10, Email: "go@lang"}},
		{name: "empty", user: User{Name: "", Age: 10, Email: "go@lang"}, wantErr: ErrEmptyName},
		{name: "age", user: User{Name: "Go", Age: 200, Email: "go@lang"}, wantErr: ErrInvalidAge},
		{name: "email", user: User{Name: "Go", Age: 10, Email: "invalid"}, wantErr: ErrInvalidEmail},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := ValidateUser(tc.user)
			if tc.wantErr == nil && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("expected %v, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidateUserJoin(t *testing.T) {
	t.Parallel()

	err := ValidateUser(User{Name: "", Age: -1, Email: "bad"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrEmptyName) || !errors.Is(err, ErrInvalidAge) || !errors.Is(err, ErrInvalidEmail) {
		t.Fatalf("expected joined error to include all sentinel errors: %v", err)
	}
}
