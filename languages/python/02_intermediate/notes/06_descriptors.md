# Descriptors：`__get__`、`__set__`、`__delete__`、property 的本质

> Descriptor 是 Python 属性访问机制的底层协议——`property`、`classmethod`、`staticmethod` 甚至 `__slots__` 都是基于 descriptor 实现的。

## Descriptor Protocol

```python
class Descriptor:
    def __get__(self, obj, objtype=None): ...
    def __set__(self, obj, value): ...
    def __delete__(self, obj): ...
```

- 只定义了 `__get__` → **non-data descriptor**（可被实例属性覆盖）
- 定义了 `__set__` 或 `__delete__` → **data descriptor**（优先于实例属性）

## 优先级链（属性查找顺序）

```
data descriptor  >  实例 __dict__  >  non-data descriptor
```

```python
class DataDesc:
    def __get__(self, obj, objtype=None):
        return "data desc"
    def __set__(self, obj, value):
        print(f"set {value}")

class NonDataDesc:
    def __get__(self, obj, objtype=None):
        return "non-data desc"

class Demo:
    dd = DataDesc()
    ndd = NonDataDesc()

d = Demo()
d.dd          # "data desc"
d.__dict__["dd"] = "shadow?"
d.dd          # 仍然是 "data desc"，data descriptor 优先级更高

d.ndd         # "non-data desc"
d.__dict__["ndd"] = "shadow!"
d.ndd         # "shadow!"，实例属性覆盖了 non-data descriptor
```

## property 就是 data descriptor

```python
class C:
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

# 等价于：
class C:
    def getx(self): return self._x
    def setx(self, v): self._x = v
    x = property(getx, setx)
```

`property()` 返回的就是一个 data descriptor 对象，`__get__` 调用 getter，`__set__` 调用 setter。

## `__set_name__`：自动知道属性名（Python 3.6+）

```python
class Field:
    def __set_name__(self, owner, name):
        self._name = name
    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return obj.__dict__.get(self._name)
    def __set__(self, obj, value):
        obj.__dict__[self._name] = value

class Model:
    name = Field()   # 自动调用 Field.__set_name__(Model, "name")
    age = Field()    # 自动调用 Field.__set_name__(Model, "age")
```

## 实战：ORM 风格的 validator

```python
class ValidatedField:
    def __init__(self, validator):
        self.validator = validator
    def __set_name__(self, owner, name):
        self._name = name
    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return obj.__dict__.get(self._name)
    def __set__(self, obj, value):
        self.validator(value)
        obj.__dict__[self._name] = value

def positive(v):
    if v <= 0:
        raise ValueError("must be positive")

class Order:
    price = ValidatedField(positive)
    quantity = ValidatedField(positive)

    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity

Order(-1, 5)  # ValueError: must be positive
```

## classmethod / staticmethod 的实现本质

```python
class classmethod:
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        return self.func.__get__(objtype)  # 绑定到类
        # 实际更复杂，有 _MethodType 包装

class staticmethod:
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        return self.func  # 原样返回，不做绑定
```

## 与 C++/Go 的对比

| 概念 | Python | C++ | Go |
|------|--------|-----|----|
| 属性拦截 | Descriptor protocol | `operator.()` / proxy | 无原生机制，用 getter/setter 方法 |
| 计算属性 | `@property` | 成员函数 | 方法，无语法糖 |
| 方法绑定 | Descriptor 自动绑定 self | `this` 隐式传入 | 方法只是第一个参数为 receiver 的函数 |
| 静态方法 | `@staticmethod` 跳过绑定 | `static` 成员函数 | 包级函数 |
| 元编程 | Descriptor + metaclass | 模板 + SFINAE + 宏 | reflection + go:generate |
