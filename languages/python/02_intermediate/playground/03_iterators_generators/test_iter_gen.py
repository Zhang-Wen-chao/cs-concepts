import pytest
from iter_gen_demo import (
    Range,
    fibonacci,
    read_in_chunks,
    flatten,
    pipeline_logs,
)


class TestRange:
    def test_iteration(self):
        r = Range(0, 3)
        assert list(r) == [0, 1, 2]

    def test_exhaustion(self):
        r = Range(0, 1)
        assert list(r) == [0]
        assert list(r) == []


class TestFibonacci:
    def test_first_10(self):
        f = fibonacci()
        assert [next(f) for _ in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


class TestChunks:
    def test_even_chunks(self):
        result = list(read_in_chunks([1, 2, 3, 4, 5, 6], 2))
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_remainder(self):
        result = list(read_in_chunks([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]


class TestFlatten:
    def test_nested_lists(self):
        assert list(flatten([1, [2, [3, 4]], 5])) == [1, 2, 3, 4, 5]

    def test_empty(self):
        assert list(flatten([])) == []


class TestPipeline:
    def test_log_pipeline(self):
        logs = [
            "ERROR 10:00 failed",
            "DEBUG 10:01 detail",
            "INFO 10:02 running",
            "",
        ]
        result = pipeline_logs(logs)
        assert len(result) == 2
        assert all(e["level"] != "DEBUG" for e in result)
