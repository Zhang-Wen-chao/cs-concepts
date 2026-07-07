import pytest
from oop_demo import (
    A, B, C,
    Singleton,
    Temperature,
    Date,
    D,
)


class TestSingleton:
    def test_same_instance(self):
        s1 = Singleton(1)
        s2 = Singleton(2)
        assert s1 is s2

    def test_single_init(self):
        s1 = Singleton(1)
        s2 = Singleton(2)
        assert s1.value == 1
        assert s2.value == 1  # 同一个实例，init 只执行一次


class TestProperty:
    def test_get_set(self):
        t = Temperature(25)
        assert t.celsius == 25
        assert t.fahrenheit == 77

    def test_setter_validation(self):
        t = Temperature(0)
        with pytest.raises(ValueError):
            t.celsius = -300


class TestClassMethod:
    def test_from_string(self):
        d = Date.from_string("2024-01-15")
        assert d.year == 2024
        assert d.month == 1
        assert d.day == 15

    def test_is_valid(self):
        assert Date.is_valid("2024-01-15")
        assert not Date.is_valid("2024-13-01")


class TestMRO:
    def test_mro_order(self):
        assert D.mro() == [D, B, C, A, object]

    def test_method_resolution(self):
        d = D()
        assert d.who() == "B"  # B 在 C 之前
