package main

import (
	"bufio"
	"fmt"
	"os"

	ingest "github.com/aaron/cs-concepts/go-projects/02_ingest_pipeline"
)

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	var entries []ingest.Entry
	for scanner.Scan() {
		entry, err := ingest.ParseLine(scanner.Text())
		if err != nil {
			fmt.Println("skip:", err)
			continue
		}
		entries = append(entries, entry)
	}
	fmt.Println("counts:", ingest.AggregateByPath(entries))
}
