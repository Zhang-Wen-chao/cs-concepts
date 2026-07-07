"""迭代器与生成器：自定义迭代器、生成器、yield from、协程基础"""


class Range:
    def __init__(self, start, stop):
        self.current = start
        self.stop = stop
    def __iter__(self):
        return self
    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def read_in_chunks(lines, chunk_size=3):
    for i in range(0, len(lines), chunk_size):
        yield lines[i:i + chunk_size]


def flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def pipeline_logs(logs):
    parsed = (parse_line(line) for line in logs if line.strip())
    filtered = (e for e in parsed if e["level"] != "DEBUG")
    return list(filtered)


def parse_line(line):
    parts = line.strip().split(" ", 2)
    return {"level": parts[0], "time": parts[1], "msg": parts[2]}
