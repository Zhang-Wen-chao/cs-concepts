import time
import pytest
from decorators_demo import timer, retry, cached, CountCalls


class TestTimer:
    def test_timer_records_duration(self):
        @timer
        def slow():
            time.sleep(0.01)
        slow()
        assert slow._last_duration >= 0.01


class TestRetry:
    def test_retry_success(self):
        attempts = [0]
        @retry(max_attempts=3)
        def flaky():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ConnectionError("timeout")
            return "ok"
        assert flaky() == "ok"
        assert attempts[0] == 2

    def test_retry_exhausted(self):
        @retry(max_attempts=2)
        def always_fails():
            raise ValueError("boom")
        with pytest.raises(ValueError):
            always_fails()


class TestCached:
    def test_cache_hits(self):
        call_count = [0]
        @cached
        def fib(n):
            call_count[0] += 1
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
        assert fib(10) == 55
        hits = call_count[0]
        call_count[0] = 0
        assert fib(10) == 55
        assert call_count[0] == 0  # cached, no new calls

    def test_cache_clear(self):
        @cached
        def add(a, b):
            return a + b
        assert add(1, 2) == 3
        add.clear()
        assert (1, 2) not in add.cache


class TestCountCalls:
    def test_counts(self):
        @CountCalls
        def greet(name):
            return f"hi {name}"
        assert greet("alice") == "hi alice"
        assert greet.count == 1
        assert greet("bob") == "hi bob"
        assert greet.count == 2
