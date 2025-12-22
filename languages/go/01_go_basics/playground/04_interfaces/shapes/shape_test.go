package shapes

import (
	"math"
	"testing"
)

func TestRectangleImplementsShape(t *testing.T) {
	t.Parallel()

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
}
