# 元类：type 是元类、\_\_new\_\_ 与 \_\_init\_\_、\_\_call\_\_、实际应用

> 元类是"类的类"——它控制类的创建行为。你 99% 不会写元类，但理解元类让你理解 Python 的类机制。

## 一切类都由 type 创建

```python
# 普通的类定义
class MyClass:
    x = 10
    def say(self):
        return "hello"

# 等价于：
MyClass = type('MyClass', (), {'x': 10, 'say': lambda self: "hello"})

print(type(MyClass))  # <class 'type'> — type 是元类
print(type(int))      # <class 'type'> — 连 int 的类也是 type
print(type(type))     # <class 'type'> — type 是自己的元类
```

**关系链**：`instance.__class__` → `class.__class__` → `type`

## 自定义元类

```python
class Meta(type):
    """一个记录类创建过程的元类"""
    def __new__(mcs, name, bases, namespace):
        print(f"Meta.__new__: 创建类 {name}")
        # 可以修改 namespace
        namespace['version'] = 1
        return super().__new__(mcs, name, bases, namespace)
    
    def __init__(cls, name, bases, namespace):
        print(f"Meta.__init__: 初始化类 {name}")
        super().__init__(name, bases, namespace)
    
    def __call__(cls, *args, **kwargs):
        print(f"Meta.__call__: 调用 {cls.__name__}({args}, {kwargs})")
        return super().__call__(*args, **kwargs)

class MyService(metaclass=Meta):
    def __init__(self, name):
        self.name = name
        print(f"实例 __init__: {name}")

s = MyService("test")
# 输出：
# Meta.__new__: 创建类 MyService
# Meta.__init__: 初始化类 MyService
# Meta.__call__: 调用 MyService(('test',), {})
# 实例 __init__: test

print(MyService.version)  # 1 — 元类注入的属性
```

**执行顺序**：`Meta.__new__` → `Meta.__init__` → `Meta.__call__` → `__new__` → `__init__`

## 实际应用：ORM、单例、注册模式

```python
# 1. ORM 风格：根据描述符定义自动生成 SQL
class ModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        if name == 'Model':
            return super().__new__(mcs, name, bases, namespace)
        
        # 自动收集字段
        fields = {}
        for key, val in list(namespace.items()):
            if isinstance(val, (int, str)):
                fields[key] = val
                namespace.pop(key)
        namespace['__table__'] = name.lower()
        namespace['__fields__'] = fields
        return super().__new__(mcs, name, bases, namespace)

class Model(metaclass=ModelMeta):
    pass

class User(Model):
    id = int
    name = str
    email = str

print(User.__table__)   # 'user'
print(User.__fields__)  # {'id': <class 'int'>, ...}
```

```python
# 2. 单例元类
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("初始化数据库连接")

db1 = Database()  # 输出 "初始化数据库连接"
db2 = Database()  # 不输出
print(db1 is db2) # True
```

```python
# 3. 注册模式（插件系统）
class RegistryMeta(type):
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if not getattr(cls, '__abstract__', False):
            mcs.registry[name] = cls
        return cls

class Plugin(metaclass=RegistryMeta):
    __abstract__ = True

class JSONPlugin(Plugin):
    pass

class XMLPlugin(Plugin):
    pass

print(RegistryMeta.registry)
# {'JSONPlugin': <class '...'>, 'XMLPlugin': <class '...'>}
```

## 应该什么时候用元类

**该用**：
- 框架/库的设计（Django ORM、SQLAlchemy）
- 需要拦截/修改类创建过程
- 注册模式、插件系统

**不该用**：
- 可以用装饰器替代
- 可以用 `__init_subclass__` 替代（PEP 487）
- 可以用类装饰器替代

```python
# PEP 487 的 __init_subclass__ 可以替代很多元类场景
class BasePlugin:
    plugins = {}
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BasePlugin.plugins[cls.__name__] = cls

class JSONPlugin(BasePlugin):
    pass

print(BasePlugin.plugins)  # {'JSONPlugin': <class '...'>}
```

## 总结

- 元类是"类的类"，`type` 是默认元类
- 自定义元类继承 `type`，重写 `__new__`/`__init__`/`__call__`
- 创建类时：元类 `__new__` → `__init__` → 实例化时：元类 `__call__`
- 实际应用：ORM 字段收集、单例、注册模式
- **不用追求写元类**——能看懂就行。大部分场景用 `__init_subclass__`、类装饰器、descriptor 就够了
