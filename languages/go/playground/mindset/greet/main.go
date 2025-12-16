package main

import (
	"flag"
	"fmt"
	"log"
)

var (
	name = flag.String("name", "gopher", "person to greet")
	lang = flag.String("lang", "zh", "language for greeting (zh|en)")
)

func main() {
	flag.Parse()

	msg, err := greeting(*name, *lang)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(msg)
}
