package shapes

import (
	"fmt"
	"math"
)

// Shape 是最小接口，用于演示鸭子类型。
type Shape interface {
	Area() float64
	Perimeter() float64
}

// Rectangle 演示结构体 + 方法接收者。
type Rectangle struct {
	Width  float64
	Height float64
}

func (r Rectangle) Area() float64      { return r.Width * r.Height }
func (r Rectangle) Perimeter() float64 { return 2 * (r.Width + r.Height) }

// Circle 展示不同类型也可实现相同接口。
type Circle struct {
	Radius float64
}

func (c Circle) Area() float64      { return math.Pi * c.Radius * c.Radius }
func (c Circle) Perimeter() float64 { return 2 * math.Pi * c.Radius }

// Describe 根据具体类型生成说明。
func Describe(s Shape) string {
	switch v := s.(type) {
	case Rectangle:
		return fmt.Sprintf("rect %.1fx%.1f area=%.1f", v.Width, v.Height, v.Area())
	case Circle:
		return fmt.Sprintf("circle r=%.1f area=%.1f", v.Radius, v.Area())
	default:
		return fmt.Sprintf("shape area=%.1f", s.Area())
	}
}
