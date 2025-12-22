package ingest

import "testing"

func TestParseLine(t *testing.T) {
	t.Parallel()

	entry, err := ParseLine("GET /login 200 15ms")
	if err != nil {
		t.Fatalf("ParseLine error: %v", err)
	}
	if entry.Path != "/login" || entry.Status != 200 || entry.Latency.String() != "15ms" {
		t.Fatalf("unexpected entry: %+v", entry)
	}
}

func TestAggregateByPath(t *testing.T) {
	t.Parallel()

	entries := []Entry{{Path: "/login"}, {Path: "/login"}, {Path: "/home"}}
	counts := AggregateByPath(entries)
	if counts["/login"] != 2 || counts["/home"] != 1 {
		t.Fatalf("unexpected counts: %#v", counts)
	}
}
