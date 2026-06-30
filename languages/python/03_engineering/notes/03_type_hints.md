# 类型注解：typing 模块、Protocol、TypedDict、Literal、泛型 Self

> Python 的类型注解不会影响运行时——但它是现代 Python 工程化的基石。

## 基础注解

```python
def greet(name: str, age: int = 30) -> str:
    return f"{name} is {age} years old"

# 运行时不影响
print(greet("Alice"))       # "Alice is 30 years old"
print(greet(42, "hi"))      # 也能跑——注解只是文档（除非用 type checker）
```

**关键理解**：Python 类型注解不会改变任何运行时行为。它是给开发者+静态检查器看的。

## Typing 模块

```python
from typing import (
    List, Dict, Tuple, Set,
    Optional, Union, Any,
    Sequence, Mapping,
    Callable, TypeVar,
)

# 基础容器
x: List[int] = [1, 2, 3]
y: Dict[str, int] = {"a": 1}
z: Tuple[int, str, float] = (1, "hi", 3.14)
s: Set[int] = {1, 2, 3}

# Optional = Union[X, None]
name: Optional[str] = None  # str | None

# Union
value: Union[int, str] = 42     # int | str（Python 3.10+ 写法）
value = "hello"

# Any（关闭类型检查）
anything: Any = "whatever"

# Callable
handler: Callable[[int, str], bool]  # (int, str) -> bool

# Sequence/Mapping（抽象容器类型）
def process(items: Sequence[int]) -> int:
    return sum(items)  # 接受 list, tuple, range
```

## Python 3.10+ 简化语法

```python
# 3.10+：| 替代 Union
x: int | str = 42
y: int | None = None  # 等价于 Optional[int]

# 3.9+：原生容器泛型（不再需要 from typing import List）
z: list[int] = [1, 2, 3]
d: dict[str, int] = {"key": 1}

# 这需要在 3.9+，如果你的项目是 3.8 还得用旧的
```

## Protocol：结构鸭子类型

```python
from typing import Protocol

# Protocol 定义结构（Go 风格 interface）
class JSONSerializable(Protocol):
    def to_json(self) -> str:
        ...

class Person:
    def to_json(self) -> str:
        return '{"name": "Alice"}'

class Car:
    def to_json(self) -> str:
        return '{"model": "Tesla"}'

class Dog:
    pass  # 没有 to_json

def serialize(obj: JSONSerializable) -> str:
    return obj.to_json()

serialize(Person())  # ✅ 类型检查通过
serialize(Car())     # ✅ 通过
# serialize(Dog())   # ❌ 类型检查报错（没有 to_json）
```

**Protocol 对比 ABC**：
- ABC：显式继承（`class MyClass(MyABC)`）
- Protocol：结构匹配（"有这个签名就行"）— 更像 Go 的 interface

## TypedDict：字典的结构化类型

```python
from typing import TypedDict

# 定义结构化的字典
class User(TypedDict):
    name: str
    age: int
    email: str

# 使用
def send_email(user: User) -> None:
    print(f"发送到 {user['email']}")

alice: User = {"name": "Alice", "age": 30, "email": "a@b.com"}
send_email(alice)

# TypedDict 不影响运行时——只是类型检查
# Python 3.8+ 可用（加上 from __future__ import annotations）
```

## Literal：精确值约束

```python
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"模式: {mode}")

set_mode("read")     # ✅
set_mode("delete")   # ❌ 类型检查报错
# set_mode(42)       # ❌ 类型不匹配

# 多个 Literal 值
type Mode = Literal["r", "w", "a", "r+"]
```

## Self 类型

```python
from typing import Self

class Builder:
    def __init__(self):
        self._items: list[int] = []
    
    def add(self, item: int) -> Self:  # ✅ 返回 Self 类型
        self._items.append(item)
        return self
    
    def build(self) -> list[int]:
        return self._items

class SpecialBuilder(Builder):
    def extra(self) -> str:
        return "special"

# Self 确保返回类型是当前类（子类也正确）
sb = SpecialBuilder().add(1).add(2)  # 类型是 SpecialBuilder
```

**没有 Self 的痛**：之前得写 `-> 'Builder'`，子类链式调用类型会变回基类。

## TypeVar：泛型

```python
from typing import TypeVar, List

T = TypeVar('T')           # 任意类型
U = TypeVar('U', int, str) # 约束为 int 或 str
V = TypeVar('V', bound=Comparable)  # 约束为 Comparable 的子类

def first(items: List[T]) -> T:
    return items[0]

print(first([1, 2, 3]))    # T → int
print(first(["a", "b"]))   # T → str
```

## pyright / mypy

```bash
# mypy（老牌，慢）
pip install mypy
mypy src/ --strict

# pyright（微软维护，快，VSCode Pylance 用）
npm install -g pyright
# 或 pip install pyright
pyright src/

# 配置文件 pyproject.toml
[tool.pyright]
typeCheckingMode = "strict"
exclude = ["tests/", "node_modules/"]
```

**严格模式建议**：项目中至少开启 `basic`，新项目可以考虑 `strict`。

## 总结

| 概念 | 说明 |
|---|---|
| 基础注解 | `def f(x: int) -> str:` — 给人和工具看 |
| `Protocol` | 结构鸭子类型，Go 风格 interface |
| `TypedDict` | 字典结构化 |
| `Literal` | 精确值约束 |
| `Self` | 方法返回当前实例类型 |
| `TypeVar` | 泛型参数 |
| 检查器 | `mypy`（慢但完善）或 `pyright`（快） |

Python 类型注解从 PEP 484 开始，3.10+ 后大幅简化。**静态类型 + 动态运行**的组合拳——既有开发期检查的收益，又有运行时灵活性的优势。
