package main

import "fmt"

var greetingTemplates = map[string]string{
	"en": "Hello, %s!",
	"zh": "你好，%s！",
}

func greeting(name, lang string) (string, error) {
	if name == "" {
		name = "gopher"
	}

	tmpl, ok := greetingTemplates[lang]
	if !ok {
		return "", fmt.Errorf("unsupported language %q", lang)
	}

	return fmt.Sprintf(tmpl, name), nil
}
