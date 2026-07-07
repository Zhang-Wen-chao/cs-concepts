"""性能分析：timeit、cProfile、line_profiler 使用示例"""


def fib_recursive(n: int) -> int:
    if n < 2:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_iterative(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def process_items(items: list[int]) -> list[int]:
    result: list[int] = []
    for item in items:
        if item % 2 == 0:
            result.append(item * 2)
        else:
            result.append(item * 3)
    return result


def compute_stats(numbers: list[float]) -> dict[str, float]:
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std = variance ** 0.5
    return {"mean": mean, "std": std}


if __name__ == "__main__":
    import timeit
    n = 30
    recursive_time = timeit.timeit(lambda: fib_recursive(n), number=10)
    iterative_time = timeit.timeit(lambda: fib_iterative(n), number=1000)
    print(f"fib_recursive({n}): {recursive_time:.4f}s (10 runs)")
    print(f"fib_iterative({n}): {iterative_time:.4f}s (1000 runs)")
