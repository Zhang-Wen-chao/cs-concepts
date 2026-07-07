"""Time-window aggregator for log statistics."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock

from log_analyzer.models import LogEntry, Stats


class Aggregator:
    """Thread-safe in-memory log aggregator with configurable time windows."""

    def __init__(self, window: timedelta = timedelta(minutes=5)) -> None:
        self.window = window
        self._lock = Lock()
        self._stats: dict[datetime, Stats] = defaultdict(Stats)

    def add(self, entry: LogEntry) -> None:
        bucket = self._bucket(entry.timestamp)
        with self._lock:
            stats = self._stats[bucket]
            stats.total_requests += 1
            stats.latencies.append(entry.latency_ms)
            stats.by_path[entry.path] = stats.by_path.get(entry.path, 0) + 1
            stats.by_status[entry.status] = stats.by_status.get(entry.status, 0) + 1
            if entry.status >= 400:
                stats.errors += 1

    def _bucket(self, ts: datetime) -> datetime:
        """Round down to window boundary."""
        ts_ts = ts.timestamp()
        bucket_ts = ts_ts - (ts_ts % self.window.total_seconds())
        return datetime.fromtimestamp(bucket_ts, tz=ts.tzinfo)

    def query(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[datetime, Stats]:
        with self._lock:
            result: dict[datetime, Stats] = {}
            for bucket, stats in self._stats.items():
                if since and bucket < since:
                    continue
                if until and bucket >= until:
                    continue
                result[bucket] = stats
            return dict(sorted(result.items()))

    def summary(self) -> Stats:
        total = Stats()
        with self._lock:
            for stats in self._stats.values():
                total.total_requests += stats.total_requests
                total.errors += stats.errors
                total.latencies.extend(stats.latencies)
                for path, count in stats.by_path.items():
                    total.by_path[path] = total.by_path.get(path, 0) + count
                for status, count in stats.by_status.items():
                    total.by_status[status] = total.by_status.get(status, 0) + count
        return total
