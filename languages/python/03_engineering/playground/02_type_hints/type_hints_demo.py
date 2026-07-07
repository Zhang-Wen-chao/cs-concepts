"""类型注解：Protocol、TypedDict、Generic、Literal"""
from typing import Protocol, TypedDict, Generic, TypeVar, Literal, overload


# === Protocol（结构鸭子类型）===
class Drawable(Protocol):
    def draw(self) -> str: ...


def render(obj: Drawable) -> str:
    return obj.draw()


class Circle:
    def draw(self) -> str:
        return "circle"


class Square:
    def draw(self) -> str:
        return "square"


# === TypedDict ===
class Movie(TypedDict):
    title: str
    year: int
    rating: float


def summarize(m: Movie) -> str:
    return f"{m['title']} ({m['year']}): {m['rating']}"


# === Generic ===
T = TypeVar("T")
U = TypeVar("U")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    def push(self, item: T) -> None:
        self._items.append(item)
    def pop(self) -> T:
        return self._items.pop()
    def is_empty(self) -> bool:
        return len(self._items) == 0


def first(seq: list[T]) -> T | None:
    return seq[0] if seq else None


# === Literal ===
def set_mode(mode: Literal["read", "write", "append"]) -> str:
    return f"mode set to {mode}"


# === overload ===
@overload
def double(x: int) -> int: ...

@overload
def double(x: str) -> str: ...

def double(x: int | str) -> int | str:
    if isinstance(x, int):
        return x * 2
    return x + x
