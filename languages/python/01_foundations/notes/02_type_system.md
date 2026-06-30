# 类型系统：可变/不可变、浅拷贝深拷贝、is vs ==、\_\_slots\_\_

## 可变 vs 不可变

```python
# 不可变：int, float, str, tuple, frozenset, bytes
x = 42
y = x
x += 1       # 创建了新对象
print(x, y)  # 43 42 — y 没变

# 可变：list, dict, set, bytearray
a = [1, 2]
b = a
a.append(3)
print(a, b)  # [1, 2, 3] [1, 2, 3] — b 跟着变了
```

**理解**：不可变类型的"修改"实际上是创建新对象+重新绑定名字。可变类型则直接修改对象内容。

```python
# 验证一下
a = [1, 2]
print(id(a))  # 某个地址
a.append(3)
print(id(a))  # 同一个地址 — 原地修改

s = "hi"
print(id(s))
s += "!"
print(id(s))  # 不同地址 — 创建了新对象
```

## is vs ==

```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b   # True — 值相等（调用 __eq__）
a is b   # False — 不同对象
a is c   # True — 同一个对象
```

**陷阱：小整数缓存**

```python
x = 256
y = 256
x is y  # True（CPython 缓存 -5 到 256 的小整数）

x = 257
y = 257
x is y  # False（超出缓存范围）
```

**规则**：比较值永远用 `==`，`is` 只用来检查 `None`、`True`、`False` 和单例。

## 浅拷贝 vs 深拷贝

```python
import copy

original = [[1, 2], [3, 4]]

# 浅拷贝：只复制外层容器
shallow = copy.copy(original)
shallow[0].append(99)
print(original)  # [[1, 2, 99], [3, 4]] — 内层被改了！

# 深拷贝：递归复制所有内容
deep = copy.deepcopy(original)  # 重新算，避开刚才的修改
deep[0].append(88)
print(original)  # [[1, 2, 99], [3, 4]] — 不受影响
```

**自定义对象的支持**：实现 `__copy__()` 和 `__deepcopy__()` 控制拷贝行为。

## `__slots__`

```python
# 默认：每个实例都有 __dict__
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
p.z = 3    # 可以动态加属性
print(p.__dict__)  # {'x': 1, 'y': 2, 'z': 3}
```

```python
# 用 __slots__ 固定属性
class SlotPoint:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

sp = SlotPoint(1, 2)
# sp.z = 3     # ❌ AttributeError
# print(sp.__dict__)  # ❌ 没有 __dict__
```

**为什么用 `__slots__`**：
1. 省内存（每个实例省几十字节的 dict overhead）
2. 属性访问更快（直接从 descriptor 取，不查 dict）

**什么时候用**：百万级实例的游戏实体、数据类。大部分业务代码不值得。

## 总结

- 可变/不可变是 Python 最基础的概念——搞错会导致隐蔽的 bug
- `==` 比较值，`is` 比较身份——永远用 `==` 除非检查 `None`
- 浅拷贝只拷贝一层，深拷贝递归全拷贝
- `__slots__` 是内存优化工具，不是常规特性
