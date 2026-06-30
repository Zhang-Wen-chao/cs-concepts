# Python 思维 — 问题版（自测用）

> 你 Python 熟，但看看这些坑你踩过几个。

## Q1

Python 函数参数默认值是 `[]` 有什么问题？

<details>
<summary>答案</summary>

```python
def add_item(item, lst=[]):
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2]  ← ❌ 不是 [2]
```

默认参数在**函数定义时只计算一次**，同一个 list 对象被所有调用共享。

**正解**：`def add_item(item, lst=None): lst = lst or []`

</details>

## Q2

Python 里 `is` 和 `==` 有什么区别？C++ 和 Go 里等价的对比是什么？

<details>
<summary>答案</summary>

- `==` 比较**值**（调用 `__eq__`）
- `is` 比较**身份**（内存地址）

```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b   # True（值一样）
a is b   # False（两个对象）
```

| 语言 | 值比较 | 身份比较 |
|---|---|---|
| Python | `==` | `is` |
| C++ | `==`（可重载） | `&a == &b`（地址比较） |
| Go | `==`（可比较的类型） | `&a == &b` 或 `unsafe.Pointer` |

</details>

## Q3

Python 的 `for` 循环里改循环变量，会影响外部吗？C++ 呢？

<details>
<summary>答案</summary>

```python
i = 0
for i in range(5):
    pass
print(i)  # 4 — Python 泄露了循环变量！
```

```cpp
int i = 0;
for (int i = 0; i < 5; i++) {}
// i 还是 0 — C++ 的循环变量有自己的作用域（C++17 后）
```

Python 2 里更明显，Python 3 里列表推导也还有泄露问题（Python 3.8 之前）。

</details>

## Q4

`try/finally` 和 `with` 语句，哪个是 Python 的 RAII？

<details>
<summary>答案</summary>

**两者都是 RAII 的变体。**

- `with open('f') as f:` — 调用 `__enter__` / `__exit__`，C++ 叫 RAII，Python 叫上下文管理器
- `try/finally` — 手动版 RAII，保证 finally 子句一定执行

C++ 靠**析构函数自动触发**，Python 靠 `with` 或 `finally` 显式保证。

实际上 `contextlib.contextmanager` 实现的就是一个 RAII 装饰器：

```python
from contextlib import contextmanager

@contextmanager
def managed_file(name):
    f = open(name, 'w')
    try:
        yield f
    finally:
        f.close()  # "析构"
```

</details>

## Q5

Python 里 `__slots__` 是做什么的？什么时候该用？

<details>
<summary>答案</summary>

```python
class Point:
    __slots__ = ('x', 'y')  # 不允许再添加其他属性
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
p.z = 3  # ❌ AttributeError
```

`__slots__` 阻止了 `__dict__` 的创建，每个实例**省掉一个字典**（大约几十字节），属性访问更快。

**用**：频繁创建大量实例的时候（比如百万级游戏对象）

**不用**：大部分业务代码 — 不值得牺牲灵活性

</details>

## Q6

Python 的 GIL 是什么？多线程 Python 到底有没有用？

<details>
<summary>答案</summary>

**GIL** = Global Interpreter Lock，保证同一时刻只有一个线程执行 Python 字节码。

- **CPU 密集型**：多线程 = 没用（一次只有一个线程在算）
- **I/O 密集型**：多线程 = 有用（GIL 在 I/O 等待时释放，别的线程可以跑）
- **替代方案**：多进程（`multiprocessing`）或 asyncio

Go：没有 GIL，goroutine 是 M:N 调度，在多核上并行跑
C++：没有 GIL，`std::thread` 直接调用 OS 线程，`std::async` 自动管理

**结论**：Python 多线程适合 I/O 场景，不适合 CPU 并行。

</details>
