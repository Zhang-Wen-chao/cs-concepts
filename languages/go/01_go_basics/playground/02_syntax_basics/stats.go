package main

import "fmt" // fmt.Errorf 用于构造带格式的错误

// Report 是一个结构体（struct）——字段名后跟类型。大写字段表示导出，可被其他包访问。
// Count: 元素个数
// Sum: 数值总和
// Avg: 平均值，使用 float64
// Classification: 对总和的分类标签（positive/zero/negative）
type Report struct {
	Count          int
	Sum            int
	Avg            float64
	Classification string
}

// describeNumbers 演示多返回值：第一个是 Report，第二个是 error。
// 签名：func describeNumbers(nums []int) (Report, error)
//   - 参数：nums 是 []int（切片类型，动态长度）。切片包含指向底层数组的指针、长度和容量。
//   - 返回：Report 汇总统计信息，error 指出是否存在“空输入”等异常。
func describeNumbers(nums []int) (Report, error) {
	if len(nums) == 0 {
		// 当输入为空时返回零值 Report 和一个错误。fmt.Errorf 类似 fmt.Sprintf 但返回 error。
		return Report{}, fmt.Errorf("empty input")
	}

	sum := 0
	// for range 语句形态：for 索引, 值 := range 切片 { ... }
	// 这里用 '_' 丢弃索引，只关注值 n。
	for _, n := range nums {
		sum += n
	}

	// len(nums) 返回切片长度 (int)。要得到浮点平均值，需把整数转换成 float64。
	avg := float64(sum) / float64(len(nums))
	class := classify(sum)

	// 结构体字面量：字段名: 值。未写的字段默认零值。
	return Report{
		Count:          len(nums),
		Sum:            sum,
		Avg:            avg,
		Classification: class,
	}, nil
}

// classify 展示 switch 的“无表达式”写法，相当于 switch true { ... }。
// 签名：func classify(sum int) string —— 单参数，返回描述字符串。
func classify(sum int) string {
	switch {
	case sum < 0:
		return "negative"
	case sum == 0:
		return "zero"
	default:
		return "positive"
	}
}
