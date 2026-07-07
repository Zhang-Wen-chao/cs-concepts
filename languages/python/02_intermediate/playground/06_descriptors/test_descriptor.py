import pytest
from descriptor_demo import (
    NonDataDesc, DataDesc, Demo,
    Positive, Order, PropertyDemo,
)


class TestNonDataDescriptor:
    def test_instance_overrides(self):
        d = Demo()
        assert d.ndd == "instance override"

    def test_access_from_class(self):
        assert Demo.ndd == "non-data value"


class TestDataDescriptor:
    def test_data_desc_priority(self):
        d = Demo()
        assert d.dd == "default"
        d.dd = "custom"
        assert d.dd == "custom"


class TestPositiveValidator:
    def test_valid_values(self):
        o = Order(100, 2)
        assert o.price == 100
        assert o.quantity == 2
        assert o.total == 200

    def test_negative_price(self):
        with pytest.raises(ValueError, match="price must be positive"):
            Order(-1, 5)

    def test_zero_quantity(self):
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(10, 0)


class TestPropertyAsDescriptor:
    def test_property_get_set(self):
        d = PropertyDemo()
        assert d.x == 0
        d.x = 42
        assert d.x == 42
