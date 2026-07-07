import pytest
import asyncio
from async_demo import (
    sequential,
    concurrent,
    with_timeout,
    rate_limited,
    producer_consumer,
)


@pytest.mark.asyncio
class TestSequential:
    async def test_sequential_execution(self):
        results = await sequential(["a", "b", "c"])
        assert len(results) == 3
        assert all("data from" in r for r in results)


@pytest.mark.asyncio
class TestConcurrent:
    async def test_concurrent_execution(self):
        results = await concurrent(["a", "b", "c"])
        assert len(results) == 3


@pytest.mark.asyncio
class TestTimeout:
    async def test_no_timeout(self):
        result = await with_timeout(asyncio.sleep(0.01), timeout=1)
        assert result is None

    async def test_timeout_hit(self):
        result = await with_timeout(asyncio.sleep(10), timeout=0.01)
        assert result == "timeout"


@pytest.mark.asyncio
class TestRateLimited:
    async def test_all_complete(self):
        results = await rate_limited(list(range(10)), qps=10)
        assert len(results) == 10


@pytest.mark.asyncio
class TestProducerConsumer:
    async def test_pattern(self):
        results = await producer_consumer()
        assert len(results) == 2
        all_items = [item for sublist in results for item in sublist]
        assert len(all_items) == 5
        assert all("item" in item for item in all_items)
