"""Descriptor 协议：data vs non-data、property 本质、validator"""


class NonDataDesc:
    def __get__(self, obj, objtype=None):
        return "non-data value"


class DataDesc:
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get("_data", "default")
    def __set__(self, obj, value):
        obj.__dict__["_data"] = value


class Demo:
    ndd = NonDataDesc()
    dd = DataDesc()

    def __init__(self):
        self.ndd = "instance override"  # 实例属性覆盖 non-data descriptor


class Positive:
    def __set_name__(self, owner, name):
        self._name = f"_{name}"
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name, None)
    def __set__(self, obj, value):
        if value <= 0:
            raise ValueError(f"{self._name[1:]} must be positive, got {value}")
        object.__setattr__(obj, self._name, value)


class Order:
    price = Positive()
    quantity = Positive()

    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity

    @property
    def total(self):
        return self.price * self.quantity


# === property 本质就是 data descriptor ===
class PropertyDemo:
    def __init__(self):
        self._x = 0
    def get_x(self):
        return self._x
    def set_x(self, value):
        self._x = value
    x = property(get_x, set_x)
