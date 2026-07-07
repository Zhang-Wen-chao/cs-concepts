from datetime import datetime, timezone
from log_analyzer.aggregator import Aggregator
from log_analyzer.models import LogEntry


def _entry(path="/test", status=200, latency=10.0):
    return LogEntry(
        timestamp=datetime.now(timezone.utc),
        level="INFO",
        method="GET",
        path=path,
        status=status,
        latency_ms=latency,
    )


class TestAggregator:
    def test_add_and_summary(self):
        agg = Aggregator()
        agg.add(_entry())
        agg.add(_entry(latency=20.0))
        agg.add(_entry(status=500, latency=30.0))
        stats = agg.summary()
        assert stats.total_requests == 3
        assert stats.errors == 1
        assert stats.p50 == 20.0
        assert stats.by_path["/test"] == 3

    def test_empty_aggregator(self):
        agg = Aggregator()
        stats = agg.summary()
        assert stats.total_requests == 0
        assert stats.error_rate == 0.0
        assert stats.p50 == 0.0

    def test_by_status(self):
        agg = Aggregator()
        agg.add(_entry(status=200))
        agg.add(_entry(status=200))
        agg.add(_entry(status=500))
        stats = agg.summary()
        assert stats.by_status[200] == 2
        assert stats.by_status[500] == 1

    def test_query_with_window(self):
        agg = Aggregator()
        agg.add(_entry())
        import time
        time.sleep(0.01)
        agg.add(_entry())
        results = agg.query()
        assert len(results) >= 1
