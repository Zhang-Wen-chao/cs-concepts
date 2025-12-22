package main // main 包是可执行程序的入口

import (
	"fmt" // fmt.Println / fmt.Printf 用于打印结果
)

/*
Usage:

	go fmt ./...   // (或 gofmt) 自动格式化当前目录及子目录
	go test ./...  // 运行 stats 示例的测试
	go run .       // 执行 main.go，触发示例逻辑

练习点：if 短语句 + for range + switch + 多返回值。
*/
func main() {
	// 切片字面量：[]int 创建整数切片，花括号内是元素列表。
	nums := []int{-3, -1, 4, 10}

	// if 语句里的 := 会先声明局部变量 report/err，再根据条件分支使用。
	if report, err := describeNumbers(nums); err != nil {
		fmt.Println("describeNumbers error:", err)
	} else {
		// fmt.Printf 使用格式占位符：%d 整数，%.2f 浮点保留两位，%s 字符串。
		fmt.Printf("count=%d sum=%d avg=%.2f class=%s\n",
			report.Count, report.Sum, report.Avg, report.Classification)
	}
}
