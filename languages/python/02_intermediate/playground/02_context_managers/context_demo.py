"""上下文管理器：__enter__/__exit__、@contextmanager、ExitStack"""
from contextlib import contextmanager, ExitStack


class ManagedFile:
    def __init__(self, path, mode="r"):
        self.path = path
        self.mode = mode
        self.file = None
    def __enter__(self):
        self.file = open(self.path, self.mode)
        return self.file
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False


@contextmanager
def managed_file(path, mode="r"):
    f = open(path, mode)
    try:
        yield f
    finally:
        f.close()


class TimerContext:
    def __init__(self):
        self.elapsed = 0.0
    def __enter__(self):
        import time
        self._start = time.perf_counter()
        return self
    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self._start
        return False


@contextmanager
def atomic_write(path):
    import tempfile, os
    dirname = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(mode="w", dir=dirname, delete=False) as tmp:
        tmp_path = tmp.name
        try:
            yield tmp
            os.replace(tmp_path, path)
        except:
            os.unlink(tmp_path)
            raise
