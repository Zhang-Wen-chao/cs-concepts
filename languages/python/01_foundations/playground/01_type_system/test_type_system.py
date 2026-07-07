import pytest
from type_system_demo import (
    demo_mutability,
    demo_is_vs_eq,
    demo_copy,
    WithSlots,
    demo_slots,
)


class TestMutability:
    def test_mutable_vs_immutable(self):
        demo_mutability()

    def test_list_identity(self):
        a = [1, 2, 3]
        b = a
        assert b is a
        assert a == b


class TestIsVsEq:
    def test_identity(self):
        a = [1, 2]
        b = a
        assert a is b

    def test_equality(self):
        a = [1, 2]
        b = [1, 2]
        assert a == b
        assert a is not b


class TestCopy:
    def test_shallow(self):
        original = {"items": [1, 2]}
        shallow = dict(original)
        original["items"].append(3)
        assert shallow["items"] == [1, 2, 3]

    def test_deep(self):
        import copy
        original = {"items": [1, 2]}
        deep = copy.deepcopy(original)
        original["items"].append(3)
        assert deep["items"] == [1, 2]


class TestSlots:
    def test_slots_work(self):
        obj = WithSlots(10, 20)
        assert obj.x == 10
        assert obj.y == 20

    def test_slots_prevent_new_attr(self):
        obj = WithSlots(1, 2)
        with pytest.raises(AttributeError):
            obj.z = 3

    def test_slots_no_dict(self):
        obj = WithSlots(1, 2)
        with pytest.raises(AttributeError):
            _ = obj.__dict__
