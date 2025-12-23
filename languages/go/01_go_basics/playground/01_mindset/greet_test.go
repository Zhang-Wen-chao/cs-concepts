package main

import "testing"

// TestGreeting 展示“表驱动测试”套路：
// 1) 定义 cases := []struct{...}{...}
// 2) 循环 t.Run 每条 case
// 3) 在循环里调用被测函数并断言
func TestGreeting(t *testing.T) {
	t.Parallel() // 顶层测试标记可并行

	// struct 字面量：字段 name/lang/...，用于描述每个测试场景。
	cases := []struct {
		name    string
		lang    string
		want    string
		wantErr bool
	}{
		{name: "Go", lang: "en", want: "Hello, Go!"},
		{name: "Gopher", lang: "zh", want: "你好，Gopher！"},
		{name: "", lang: "en", want: "Hello, gopher!"},
		{name: "", lang: "zz", wantErr: true},
	}

	for _, c := range cases {
		c := c // 复制当前 case，避免闭包捕获同一个地址
		// 子测试名称由语言/名字组成，便于失败时定位
		t.Run(c.lang+"/"+c.name, func(t *testing.T) {
			t.Parallel() // 子测试也用 t.Parallel，互不阻塞

			got, err := greeting(c.name, c.lang)
			if c.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got != c.want {
				t.Fatalf("greeting(%q,%q)=%q want %q", c.name, c.lang, got, c.want)
			}
		})
	}
}
