"""Log line parser supporting multiple formats."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Iterator

from log_analyzer.models import LogEntry, LogLevel


# nginx combined: 127.0.0.1 - - [07/Jul/2026:12:00:00 +0000] "GET /api HTTP/1.1" 200 1234 "-" "curl/7.0"
_COMBINED_PATTERN = re.compile(
    r'^[\d.]+ - - \[(.+?)\] '
    r'"(\w+) (\S+) \S+" '
    r'(\d{3}) (\d+) '
    r'".*?" ".*?"'
)
_COMBINED_DT_FMT = "%d/%b/%Y:%H:%M:%S %z"


def parse_line(line: str, fmt: str = "auto") -> LogEntry | None:
    line = line.strip()
    if not line:
        return None
    if fmt == "auto":
        if line.startswith("{"):
            return _parse_json(line)
        return _parse_combined(line)
    if fmt == "json":
        return _parse_json(line)
    return _parse_combined(line)


def _parse_combined(line: str) -> LogEntry | None:
    m = _COMBINED_PATTERN.match(line)
    if not m:
        return None
    ts_str, method, path, status_str, _size = m.groups()
    ts = datetime.strptime(ts_str, _COMBINED_DT_FMT)
    status = int(status_str)
    level: LogLevel = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
    return LogEntry(
        timestamp=ts,
        level=level,
        method=method,
        path=path,
        status=status,
        latency_ms=0.0,
        body=line,
    )


def _parse_json(line: str) -> LogEntry | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    ts_str = obj.get("time") or obj.get("timestamp") or ""
    ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
    return LogEntry(
        timestamp=ts,
        level=obj.get("level", "INFO"),
        method=obj.get("method", "GET"),
        path=obj.get("path", "/"),
        status=obj.get("status", 200),
        latency_ms=obj.get("latency_ms", 0.0),
        body=line,
    )


def parse_lines(lines: list[str], fmt: str = "auto") -> list[LogEntry]:
    return [e for line in lines if (e := parse_line(line, fmt)) is not None]


def parse_stream(stream: Iterator[str], fmt: str = "auto") -> Iterator[LogEntry]:
    for line in stream:
        entry = parse_line(line, fmt)
        if entry is not None:
            yield entry
