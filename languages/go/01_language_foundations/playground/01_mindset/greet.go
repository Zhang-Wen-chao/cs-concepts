package main

import "fmt" // fmt.Sprintf 用于根据模板填充字符串

// greetingTemplates 是 map 字面量：map[键类型]值类型{ ... }。
// 这里键是语言代码（string），值是格式化模板。
// %s 是 fmt 的“字符串占位符”，后续会用名字替换。
var greetingTemplates = map[string]string{
	"en": "Hello, %s!",
	"zh": "你好，%s！",
}

// greeting 返回 (字符串, error) 两个值，体现 Go 惯常的“结果 + 错误”模式。
// 签名：func greeting(name, lang string) (string, error)
//   - 参数：name 与 lang 均为 string；第 1 个参数是人名，第 2 个是语言代码。
//   - 返回：第 1 个 string 是问候语，第 2 个 error 表示是否出错（nil 代表成功）。
func greeting(name, lang string) (string, error) {
	if name == "" {
		name = "gopher" // 如果 CLI 没传 --name 或传空串，就使用默认昵称
	}

	// 查 map：greetingTemplates[lang] 会返回值与是否存在的布尔值（“comma ok”。
	tmpl, ok := greetingTemplates[lang]
	if !ok {
		// fmt.Errorf 创建格式化错误，%q 表示带引号的字符串
		return "", fmt.Errorf("unsupported language %q", lang)
	}

	// fmt.Sprintf 根据模板插入 name，返回完整字符串
	return fmt.Sprintf(tmpl, name), nil
}
