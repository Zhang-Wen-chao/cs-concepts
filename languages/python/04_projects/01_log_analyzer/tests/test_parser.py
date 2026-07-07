from datetime import datetime
from log_analyzer.parser import parse_line, parse_lines


class TestCombinedFormat:
    def test_parse_valid_line(self):
        line = '192.168.1.1 - - [07/Jul/2026:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234 "-" "curl/7.0"'
        entry = parse_line(line)
        assert entry is not None
        assert entry.method == "GET"
        assert entry.path == "/api/users"
        assert entry.status == 200
        assert entry.level == "INFO"

    def test_parse_error_line(self):
        line = '10.0.0.1 - - [07/Jul/2026:12:01:00 +0000] "POST /api/data HTTP/1.1" 500 0 "-" "agent"'
        entry = parse_line(line)
        assert entry is not None
        assert entry.status == 500
        assert entry.level == "ERROR"

    def test_parse_bad_line(self):
        assert parse_line("not a log line") is None
        assert parse_line("") is None


class TestJsonFormat:
    def test_parse_valid_json(self):
        line = '{"time": "2026-07-07T12:00:00", "method": "GET", "path": "/health", "status": 200, "latency_ms": 5.2}'
        entry = parse_line(line, fmt="json")
        assert entry is not None
        assert entry.method == "GET"
        assert entry.status == 200
        assert entry.latency_ms == 5.2

    def test_parse_bad_json(self):
        assert parse_line("{bad json}", fmt="json") is None

    def test_auto_detect_json(self):
        line = '{"level": "ERROR", "path": "/fail"}'
        entry = parse_line(line)
        assert entry is not None
        assert entry.level == "ERROR"


class TestParseLines:
    def test_parse_multiple(self):
        lines = [
            '1.1.1.1 - - [07/Jul/2026:12:00:00 +0000] "GET /a HTTP/1.1" 200 1 "-" "-"',
            '2.2.2.2 - - [07/Jul/2026:12:00:01 +0000] "POST /b HTTP/1.1" 500 0 "-" "-"',
            "",
            "also bad",
        ]
        entries = parse_lines(lines)
        assert len(entries) == 2
