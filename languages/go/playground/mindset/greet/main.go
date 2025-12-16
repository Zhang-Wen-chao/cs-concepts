package main // main 包 + main 函数是 Go 可执行程序的唯一入口

import (
	"flag" // 解析命令行 flag（--name、--lang 等）
	"fmt"  // 格式化输出（Println/Printf）
	"log"  // 简单日志，Fatal 会输出后退出
)

/*
Usage:

	gofmt -w .                      // 自动格式化全部源码
	go test ./...                   // 运行所有单元测试
	go run . --name=Go --lang=en    // 带上 flag 运行 demo

练习点：flag 解析 + 表驱动测试；默认语言 zh，可通过 --lang=en 切换英文。
*/
// var (...) 声明一个变量块。括号允许我们在同一位置列出多个包级变量。
var (
	// flag.String(name, default, help)
	// 1) 参数名 --name，CLI 输入 `--name=Go`
	// 2) 默认值 "gopher"
	// 3) 帮助信息 `go run . --help` 时展示
	name = flag.String("name", "gopher", "person to greet")

	// 同上，lang 控制语言（zh/en）
	lang = flag.String("lang", "zh", "language for greeting (zh|en)")
)

func main() {
	flag.Parse() // 解析 CLI 传入的 --name / --lang 参数

	// := 是“短变量声明”：在函数内新建变量并赋初始值。
	// greeting(*name, *lang) 调用函数；*name 解引用 flag.String 返回的 *string，得到实际值。
	msg, err := greeting(*name, *lang)
	if err != nil {
		log.Fatal(err) // CLI 中出错直接终止并打印错误
	}

	fmt.Println(msg) // 正常情况下输出问候语
}
