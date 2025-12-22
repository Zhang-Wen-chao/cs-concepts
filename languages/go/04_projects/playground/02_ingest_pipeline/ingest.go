package ingest

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Entry 代表一条解析后的日志。
type Entry struct {
	Path    string
	Status  int
	Latency time.Duration
}

// ParseLine 解析形如 "GET /login 200 15ms" 的日志。
func ParseLine(line string) (Entry, error) {
	fields := strings.Fields(line)
	if len(fields) != 4 {
		return Entry{}, fmt.Errorf("invalid line: %q", line)
	}
	status, err := strconv.Atoi(fields[2])
	if err != nil {
		return Entry{}, fmt.Errorf("status: %w", err)
	}
	latency, err := time.ParseDuration(fields[3])
	if err != nil {
		return Entry{}, fmt.Errorf("latency: %w", err)
	}
	return Entry{Path: fields[1], Status: status, Latency: latency}, nil
}

// AggregateByPath 统计每个 path 的请求数。
func AggregateByPath(entries []Entry) map[string]int {
	counts := make(map[string]int, len(entries))
	for _, e := range entries {
		counts[e.Path]++
	}
	return counts
}
