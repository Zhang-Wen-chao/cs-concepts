"""Data models with type annotations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
LogFormat = Literal["auto", "combined", "json", "raw"]


@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    method: str
    path: str
    status: int
    latency_ms: float
    body: str = ""


@dataclass
class Stats:
    total_requests: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)
    by_path: dict[str, int] = field(default_factory=dict)
    by_status: dict[int, int] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.errors / self.total_requests

    @property
    def p50(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lats = sorted(self.latencies)
        return sorted_lats[len(sorted_lats) // 2]

    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lats = sorted(self.latencies)
        idx = int(len(sorted_lats) * 0.99)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]
