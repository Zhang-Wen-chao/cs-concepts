import pytest
from errors_demo import (
    ConfigError,
    ConfigNotFoundError,
    ConfigParseError,
    load_config,
    parse_config,
    divide_with_else,
    clean_filenames,
)


class TestCustomException:
    def test_hierarchy(self):
        assert issubclass(ConfigNotFoundError, ConfigError)
        assert issubclass(ConfigParseError, ConfigError)

    def test_error_message(self):
        err = ConfigNotFoundError("/etc/app.conf")
        assert "not found" in str(err)
        assert "/etc/app.conf" in str(err)


class TestExceptionChain:
    def test_chain_on_file_not_found(self, tmp_path):
        path = tmp_path / "nonexistent.conf"
        with pytest.raises(ConfigNotFoundError) as exc_info:
            load_config(str(path))
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    def test_parse_error(self, tmp_path):
        path = tmp_path / "bad.conf"
        path.write_text("no braces")
        exc = None
        try:
            load_config(str(path))
        except ConfigParseError as e:
            exc = e
        assert exc is not None


class TestElseFinally:
    def test_success_path(self):
        result = divide_with_else(10, 2)
        assert result["success"]
        assert result["result"] == 5

    def test_exception_path(self):
        result = divide_with_else(10, 0)
        assert not result["success"]

    def test_finally_always_called(self):
        result = divide_with_else(10, 2)
        assert result["called"]
        result = divide_with_else(10, 0)
        assert result["called"]


class TestSuppress:
    def test_skip_invalid(self):
        names = ["1", "bad", "3", "oops"]
        result = clean_filenames(names)
        assert result == [1, 3]

    def test_all_valid(self):
        assert clean_filenames(["1", "2"]) == [1, 2]
