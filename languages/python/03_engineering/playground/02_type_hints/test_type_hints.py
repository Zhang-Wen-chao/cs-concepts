import pytest
from type_hints_demo import (
    render, Circle, Square,
    summarize, Movie,
    Stack, first,
    set_mode, double,
)


class TestProtocol:
    def test_circle(self):
        assert render(Circle()) == "circle"

    def test_square(self):
        assert render(Square()) == "square"

    def test_protocol_structural(self):
        class Triangle:
            def draw(self) -> str:
                return "triangle"
        assert render(Triangle()) == "triangle"


class TestTypedDict:
    def test_summarize(self):
        m: Movie = {"title": "Inception", "year": 2010, "rating": 8.8}
        result = summarize(m)
        assert "Inception" in result
        assert "2010" in result


class TestGeneric:
    def test_stack_int(self):
        s: Stack[int] = Stack()
        s.push(1)
        s.push(2)
        assert s.pop() == 2
        assert s.pop() == 1
        assert s.is_empty()

    def test_stack_str(self):
        s: Stack[str] = Stack()
        s.push("a")
        s.push("b")
        assert s.pop() == "b"

    def test_first(self):
        assert first([1, 2, 3]) == 1
        assert first([]) is None


class TestLiteral:
    def test_valid_modes(self):
        assert set_mode("read") == "mode set to read"
        assert set_mode("write") == "mode set to write"

    def test_any_string_works_at_runtime(self):
        # Literal only checked statically by mypy/pyright, not at runtime
        assert set_mode("invalid") == "mode set to invalid"


class TestOverload:
    def test_double_int(self):
        assert double(3) == 6

    def test_double_str(self):
        assert double("ab") == "abab"
