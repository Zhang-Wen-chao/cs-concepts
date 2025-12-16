package main

import "testing"

func TestGreeting(t *testing.T) {
	t.Parallel()

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
		c := c
		t.Run(c.lang+"/"+c.name, func(t *testing.T) {
			t.Parallel()

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
