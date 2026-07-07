"""pytest 实践：fixture、parametrize、mock"""
import pytest
from unittest.mock import Mock, patch


# === Fixture ===
@pytest.fixture
def sample_data():
    return {"a": 1, "b": 2, "c": 3}


@pytest.fixture
def db_connection(tmp_path):
    db_path = tmp_path / "test.db"
    db_path.write_text("connected")
    yield db_path
    if db_path.exists():
        db_path.unlink()


# === Parametrize ===
def is_even(n):
    return n % 2 == 0


@pytest.mark.parametrize("n,expected", [
    (2, True),
    (3, False),
    (0, True),
    (-1, False),
])
def test_is_even(n, expected):
    assert is_even(n) == expected


# === Combined parametrize (cartesian product) ===
@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", [3, 4])
def test_cartesian(a, b):
    assert a + b == b + a


# === Mock ===
def fetch_data(api_client, url):
    resp = api_client.get(url)
    if resp["status"] == 200:
        return resp["data"]
    return None


def test_fetch_data():
    mock_client = Mock()
    mock_client.get.return_value = {"status": 200, "data": "hello"}
    result = fetch_data(mock_client, "http://example.com")
    assert result == "hello"
    mock_client.get.assert_called_once_with("http://example.com")


def test_fetch_data_error():
    mock_client = Mock()
    mock_client.get.return_value = {"status": 500}
    result = fetch_data(mock_client, "http://example.com")
    assert result is None


# === Fixture with teardown ===
@pytest.fixture
def mock_env():
    with patch.dict("os.environ", {"MY_VAR": "test_value"}):
        yield


def test_env(mock_env):
    import os
    assert os.environ["MY_VAR"] == "test_value"
