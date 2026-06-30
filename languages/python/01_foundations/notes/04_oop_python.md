# 面向对象：\_\_init\_\_ vs \_\_new\_\_、property、@staticmethod/@classmethod、MRO

## `__new__` vs `__init__`

```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        print(f"__new__: 创建实例, cls={cls}")
        instance = super().__new__(cls)  # 真正分配内存
        return instance
    
    def __init__(self, value):
        print(f"__init__: 初始化实例, value={value}")
        self.value = value

obj = MyClass(42)
# 输出：
# __new__: 创建实例, cls=<class '__main__.MyClass'>
# __init__: 初始化实例, value=42
```

**理解**：
- `__new__` 是**真正创建对象**的类方法（静态方法），返回实例
- `__init__` 是**初始化已创建的对象**，不能返回值（必须返回 None）
- `__new__` 先调用，`__init__` 后调用

**实际用途**：实现单例、不可变类型子类（如 int、str 的子类）。

```python
# 单例模式
class Singleton:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

## Property：可控的属性访问

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("不能低于绝对零度")
        self._celsius = value

t = Temperature(25)
print(t.fahrenheit)   # 77.0 — 像属性一样读
t.fahrenheit = 100    # 像属性一样写
print(t.celsius)      # 37.78
```

**对比**：
- Python：`obj.prop` → property descriptor → getter/setter
- C++：`obj.getProp()` / `obj.setProp(val)` → 显式方法调用
- Go：不鼓励 getter/setter，直接用字段

## @staticmethod vs @classmethod

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_str):
        """类方法：接收 cls，可以当替代构造器"""
        y, m, d = map(int, date_str.split('-'))
        return cls(y, m, d)  # 返回 cls 的实例
    
    @staticmethod
    def is_valid(date_str):
        """静态方法：不接收 cls/self，就是个普通函数"""
        try:
            y, m, d = map(int, date_str.split('-'))
            return 1 <= m <= 12 and 1 <= d <= 31
        except (ValueError, TypeError):
            return False

d = Date.from_string("2026-06-30")  # 类方法构造
print(Date.is_valid("2026-13-01"))   # False — 静态方法
```

| 类型 | 第一个参数 | 能干嘛 |
|---|---|---|
| 实例方法 | `self` | 访问实例 + 类状态 |
| `@classmethod` | `cls` | 只能访问类状态，可以做替代构造器 |
| `@staticmethod` | 没有 | 就是个类命名空间下的普通函数 |

## MRO（Method Resolution Order — 方法解析顺序）：C3 线性化

```python
class A:
    def method(self): print("A")

class B(A):
    def method(self): print("B")

class C(A):
    def method(self): print("C")

class D(B, C):
    pass

d = D()
d.method()          # B — 搜索顺序：D → B → C → A
print(D.__mro__)    # D -> B -> C -> A -> object
```

**MRO 规则**（C3 线性化 — 保证单调性的 MRO 算法）：
1. 子类优先于父类
2. 从左到右
3. 只有所有父类都满足的才是合法的

**`super()` 遵循 MRO**，不是简单地调父类：

```python
class A:
    def who(self): print("A")

class B(A):
    def who(self):
        print("B")
        super().who()  # 实际上调用谁取决于 MRO

class C(A):
    def who(self):
        print("C")
        super().who()

class D(B, C):
    def who(self):
        print("D")
        super().who()

D().who()
# D → B → C → A
```

这被称为"协作式多继承"，是 Python 菱形继承问题的解法。

## 总结

- `__new__` 创建对象，`__init__` 初始化对象 — 99% 只用 `__init__`
- `@property` 把方法伪装成属性，让接口平滑演进
- `@classmethod` 做替代构造器，`@staticmethod` 只是命名空间函数
- MRO = C3 线性化，`super()` 沿着 MRO 链调用，不是直接调父类
