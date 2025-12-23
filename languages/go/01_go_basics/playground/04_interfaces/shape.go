package shapes // 练习“组合 + 接口 + type switch”

/*
Usage:

	# 在 playground/04_interfaces 下执行
	go fmt ./...                     # 统一格式
	go test ./...                    # 跑全部测试
	go test . -run TestCircleDescribe # 只验证 Describe
*/

import (
	"fmt"
	"math"
)

// Shape 是最小接口，用于演示“只依赖需要的方法”这一设计理念。
type Shape interface {
	Area() float64
	Perimeter() float64
}

// Rectangle 演示结构体 + 方法接收者。大写导出字段，方便在其他包复用。
type Rectangle struct {
	Width  float64
	Height float64
}

// Area 计算矩形面积。
// 签名：func (r Rectangle) Area() float64 —— 接收者 r 为值类型（调用时会复制），无额外参数，返回面积。
func (r Rectangle) Area() float64 { return r.Width * r.Height }

// Perimeter 计算矩形周长。
// 签名：func (r Rectangle) Perimeter() float64 —— 同样是值接收者，返回 float64。
func (r Rectangle) Perimeter() float64 { return 2 * (r.Width + r.Height) }

// Circle 展示不同类型也可实现相同接口：无需显式 `implements`，只要方法集满足即可。
type Circle struct {
	Radius float64
}

// Area 计算圆形面积，使用 math.Pi。
// 签名：func (c Circle) Area() float64 —— 接收者为值类型，返回单个 float64。
func (c Circle) Area() float64 { return math.Pi * c.Radius * c.Radius }

// Perimeter 计算圆周长。
// 签名：func (c Circle) Perimeter() float64 —— 同样无参数，返回周长。
func (c Circle) Perimeter() float64 { return 2 * math.Pi * c.Radius }

// Describe 根据具体类型生成说明，顺便演示 type switch / type assertion。
// 签名：func Describe(s Shape) string —— 参数 s 是接口类型 Shape，返回描述文本。
func Describe(s Shape) string {
	switch v := s.(type) {
	case Rectangle:
		// fmt.Sprintf 用格式化字符串拼接信息，%.1f 表示保留 1 位小数。
		return fmt.Sprintf("rect %.1fx%.1f area=%.1f", v.Width, v.Height, v.Area())
	case Circle:
		return fmt.Sprintf("circle r=%.1f area=%.1f", v.Radius, v.Area())
	default:
		// type switch 会自动匹配不到 default，这里退回 Shape 接口提供的方法。
		return fmt.Sprintf("shape area=%.1f", s.Area())
	}
}
