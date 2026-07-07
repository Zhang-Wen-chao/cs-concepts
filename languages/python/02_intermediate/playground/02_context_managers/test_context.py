import time
import pytest
from context_demo import ManagedFile, TimerContext, atomic_write


class TestManagedFile:
    def test_read_file(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        with ManagedFile(str(path)) as f:
            assert f.read() == "hello"

    def test_file_closed(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        mf = ManagedFile(str(path))
        with mf as f:
            pass
        assert f.closed


class TestTimerContext:
    def test_times(self):
        with TimerContext() as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01


class TestAtomicWrite:
    def test_writes_atomically(self, tmp_path):
        path = tmp_path / "data.txt"
        with atomic_write(str(path)) as f:
            f.write("hello world")
        assert path.read_text() == "hello world"

    def test_rollback_on_error(self, tmp_path):
        path = tmp_path / "data.txt"
        if path.exists():
            path.write_text("original")
        try:
            with atomic_write(str(path)) as f:
                f.write("new")
                raise ValueError("rollback")
        except ValueError:
            pass
        assert not path.exists()
