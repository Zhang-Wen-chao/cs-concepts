"""异步编程：协程、Task、超时、并发的正确模式"""
import asyncio


async def fetch(session, url, delay=0):
    await asyncio.sleep(delay)
    return f"data from {url}"


async def sequential(urls):
    results = []
    for url in urls:
        data = await fetch(None, url)
        results.append(data)
    return results


async def concurrent(urls):
    async def fetch_one(url):
        return await fetch(None, url)
    tasks = [asyncio.create_task(fetch_one(url)) for url in urls]
    return await asyncio.gather(*tasks)


async def with_timeout(coro, timeout=1):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return "timeout"


async def rate_limited(urls, qps=5):
    semaphore = asyncio.Semaphore(qps)
    async def bounded_fetch(url):
        async with semaphore:
            return await fetch(None, url, delay=0.01)
    tasks = [asyncio.create_task(bounded_fetch(url)) for url in urls]
    return await asyncio.gather(*tasks)


async def producer_consumer():
    queue = asyncio.Queue()

    async def producer(n):
        for i in range(n):
            await queue.put(f"item-{i}")
            await asyncio.sleep(0.01)
        for _ in range(2):
            await queue.put(None)

    async def consumer(name):
        results = []
        while True:
            item = await queue.get()
            if item is None:
                break
            results.append(f"{name}:{item}")
        return results

    producers = [asyncio.create_task(producer(5))]
    consumers = [asyncio.create_task(consumer(f"w{i}")) for i in range(2)]
    await asyncio.gather(*producers)
    all_results = await asyncio.gather(*consumers)
    return all_results
