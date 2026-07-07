import pytest
from metaclass_demo import (
    Foo, Bar,
    ModelMeta, Model, User, Field,
    PluginBase, LogPlugin, MetricsPlugin,
)


class TestTypeAsMetaclass:
    def test_equivalent(self):
        assert type(Foo) is type
        assert type(Bar) is type
        assert Foo.__name__ == "Foo"
        assert Bar.__name__ == "Bar"

    def test_dynamic_class(self):
        obj = Bar()
        assert isinstance(obj, Bar)


class TestCustomMetaclass:
    def test_fields_collected(self):
        assert hasattr(User, "_fields")
        assert "name" in User._fields
        assert "age" in User._fields

    def test_field_instance(self):
        assert isinstance(User._fields["name"], Field)

    def test_metaclass_type(self):
        assert type(User) is ModelMeta
        assert type(Model) is ModelMeta


class TestInitSubclass:
    def test_plugins_registered(self):
        assert "LogPlugin" in PluginBase.registry
        assert "MetricsPlugin" in PluginBase.registry
        assert PluginBase.registry["LogPlugin"] is LogPlugin
