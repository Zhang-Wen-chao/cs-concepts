"""元类：type 作为元类、自定义元类、__init_subclass__"""


# === type 就是元类 ===
class Foo:
    pass

# 等价于
Bar = type("Bar", (), {})


# === 自定义元类：ORM 风格的字段收集 ===
class Field:
    def __init__(self, default=None):
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name


class ModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        fields = {}
        for key, value in list(namespace.items()):
            if isinstance(value, Field):
                fields[key] = value
        cls = super().__new__(mcs, name, bases, namespace)
        cls._fields = fields
        return cls


class Model(metaclass=ModelMeta):
    pass


class User(Model):
    name = Field(default="")
    age = Field(default=0)


# === __init_subclass__（替代大部分元类场景） ===
class PluginBase:
    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        if name.endswith("Plugin"):
            PluginBase.registry[name] = cls


class LogPlugin(PluginBase):
    pass

class MetricsPlugin(PluginBase):
    pass
