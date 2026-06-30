# 函数深入：参数传递、默认参数陷阱、\*args/\*\*kwargs、闭包、LEGB

## 参数传递机制：传递的是引用（不是值也不是引用本身）

```python
def modify(x, items):
    x += 1           # 不可变类型：x 重新绑定到新对象
    items.append(4)  # 可变类型：items 指向的对象原地修改
    items = [100]    # 重新绑定，不影响外部

a = 10
lst = [1, 2, 3]
modify(a, lst)
print(a)      # 10 — 没变（int 不可变）
print(lst)    # [1, 2, 3, 4] — 变了（list 可变，append 修改了对象）
```

**一句话**：Python 的参数传递是"传对象引用"（pass-by-object-reference）。函数内部重新绑定参数名不会影响外部，但修改可变对象会。

对比 C++：`void func(int x)` 是传值，`void func(int& x)` 是传引用。Python 更像 C++ 的 `void func(int* x)` 但解除了指针语法。

## 默认参数陷阱

```python
# ❌ 经典陷阱
def add_item(item, lst=[]):
    lst.append(item)
    return lst

print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] — 不是 [2]！
print(add_item(3))  # [1, 2, 3]

# 查看默认值
print(add_item.__defaults__)  # ([1, 2, 3],)
```

**原因**：默认参数在**函数定义时**计算一次，同一个 list 对象被所有调用共享。

```python
# ✅ 正解：用 None 做哨兵
def add_item(item, lst=None):
    lst = lst or []
    lst.append(item)
    return lst
```

## `*args` 和 `**kwargs`

```python
def log(level, *args, **kwargs):
    print(f"[{level}]", *args)
    if kwargs:
        for k, v in kwargs.items():
            print(f"  {k}={v}")

log("INFO", "hello", "world", user="alice", retry=3)
# [INFO] hello world
#   user=alice
#   retry=3
```

```python
# 解包操作符：* 和 **
def vector_len(x, y, z):
    return (x**2 + y**2 + z**2)**0.5

coords = [1, 2, 3]
print(vector_len(*coords))        # 3.742

params = {"x": 1, "y": 2, "z": 3}
print(vector_len(**params))       # 同上
```

**顺序规则**：`def f(pos, *args, default=42, **kwargs):`

## 作用域：LEGB 规则

Local → Enclosing → Global → Built-in

```python
x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)  # local
    
    inner()
    print(x)  # enclosing

outer()
print(x)  # global
```

```python
# nonlocal 和 global
count = 0

def outer():
    x = 0
    
    def inner():
        global count  # 修改全局
        nonlocal x    # 修改外层
        count += 1
        x += 1
```

**Python 没有块级作用域**：`if`/`for`/`while` 不创建新作用域（和 C++/Go 不同）。

## 闭包

```python
def make_counter(start=0):
    count = [start]  # 用 list 实现可变捕获
    def counter():
        count[0] += 1
        return count[0]
    return counter

c = make_counter(10)
print(c())  # 11
print(c())  # 12

# Python 3 后可以用 nonlocal
def make_counter_v2(start=0):
    count = start
    def counter():
        nonlocal count
        count += 1
        return count
    return counter
```

**闭包记住的是变量引用，不是值**。`nonlocal` 是 Python 3 才有的特性。

## 总结

| 概念 | 要点 |
|---|---|
| 参数传递 | 传对象引用，可变对象注意副作用 |
| 默认参数 | 只在定义时计算一次，用 None 做哨兵 |
| `*args`/`**kwargs` | 灵活的参数传递，解包符号 `*` `**` |
| LEGB | Python 唯一的作用域规则，没有块级作用域 |
| 闭包 | 捕获变量的函数 + `nonlocal` 才能修改捕获变量 |
