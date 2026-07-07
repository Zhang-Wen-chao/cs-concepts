"""类型系统演示：可变 vs 不可变、is vs ==、深浅拷贝、__slots__"""

# === 可变 vs 不可变 ===
def demo_mutability():
    # 不可变：int, str, tuple, frozenset
    a = 256
    b = 256
    assert a is b  # CPython 小整数缓存 -5 ~ 256

    # 可变对象：每次创建新对象
    a = [1, 2, 3]
    b = [1, 2, 3]
    assert a is not b  # 不同对象
    assert a == b      # 但值相等

    # 可变：list, dict, set
    x = [1, 2, 3]
    y = [1, 2, 3]
    assert x is not y
    assert x == y


# === is vs == ===
def demo_is_vs_eq():
    a = [1, 2, 3]
    b = a
    c = a[:]
    assert a is b       # 同一对象
    assert a is not c   # 不同对象
    assert a == c       # 值相同
    assert id(a) == id(b)
    assert id(a) != id(c)


# === 浅拷贝 vs 深拷贝 ===
import copy

def demo_copy():
    original = {"key": [1, 2, [3, 4]]}
    shallow = copy.copy(original)
    deep = copy.deepcopy(original)

    original["key"][2][0] = 99
    assert shallow["key"][2][0] == 99   # 共享内部对象
    assert deep["key"][2][0] == 3       # 完全独立


# === __slots__ ===
class WithSlots:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y

def demo_slots():
    obj = WithSlots(1, 2)
    assert obj.x == 1
    assert obj.y == 2
    try:
        obj.z = 3
        assert False, "should raise AttributeError"
    except AttributeError:
        pass
