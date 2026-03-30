package shapes

import (
	"math"
	"testing"
)

// 小贴士：`go test . -run Rectangle` 只跑满足条件的测试；缺省 `go test ./...` 会执行整个包。

func TestRectangleImplementsShape(t *testing.T) {
	t.Parallel()

	// 通过空白标识符断言“接口实现”——若 Rectangle 缺少 Shape 方法，编译器会报错。
	var _ Shape = Rectangle{}

	r := Rectangle{Width: 3, Height: 4}
	if r.Area() != 12 {
		t.Fatalf("area mismatch: %v", r.Area())
	}
	if r.Perimeter() != 14 {
		t.Fatalf("perimeter mismatch: %v", r.Perimeter())
	}
}

func TestCircleDescribe(t *testing.T) {
	t.Parallel()

	c := Circle{Radius: 1}
	if math.Abs(c.Area()-math.Pi) > 1e-6 {
		t.Fatalf("area mismatch: %v", c.Area())
	}

	got := Describe(c)
	if got == "" || got[:6] != "circle" {
		t.Fatalf("unexpected description: %s", got)
	}

	// type switch default 分支也应可覆盖“未知形状”，这里不单独演示，交给读者练习。
}
