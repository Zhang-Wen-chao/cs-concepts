import pytest
from functions_demo import (
    append_to,
    bad_append_to,
    sum_all,
    create_url,
    make_counter,
    make_multiplier,
    make_functions_bad,
    make_functions_good,
)


class TestDefaultArgs:
    def test_safe_default(self):
        r1 = append_to(1)
        r2 = append_to(2)
        assert r1 == [1]
        assert r2 == [2]

    def test_mutable_default_trap(self):
        # Each call to bad_append_to mutates the shared default list
        bad_append_to(1)
        bad_append_to(2)
        # 默认 list 被累积修改
        assert bad_append_to(3) == [1, 2, 3]


class TestArgsKwargs:
    def test_args(self):
        assert sum_all(1, 2, 3) == 6
        assert sum_all() == 0

    def test_kwargs(self):
        result = create_url(name="test", page="1")
        assert result == "name=test&page=1"


class TestClosure:
    def test_counter(self):
        c = make_counter()
        assert c() == 1
        assert c() == 2
        assert c() == 3

    def test_multiplier(self):
        double = make_multiplier(2)
        assert double(3) == 6

    def test_closure_trap(self):
        funcs = make_functions_bad()
        assert [f() for f in funcs] == [2, 2, 2]

    def test_closure_trap_fixed(self):
        funcs = make_functions_good()
        assert [f() for f in funcs] == [0, 1, 2]
