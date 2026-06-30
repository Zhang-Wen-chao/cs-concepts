# Python 思维：动态类型、一切皆对象、duck typing

> 你是 Python 熟手。这篇文章用对比的方式系统化你已经知道的东西。

## 动态类型 vs 静态类型

```python
# Python：运行时才知道类型
x = 42
x = "hello"   # 完全合法
x = [1, 2, 3] # 依然合法
```

```go
// Go：编译时确定类型
var x int = 42
x = "hello" // ❌ 编译错误
```

```cpp
// C++：同上，静态类型
int x = 42;
x = "hello"; // ❌ 编译错误
```

**关键区别**：静态类型帮编译器抓 bug，动态类型给你灵活性——代价是运行时才能发现类型错误。

## 一切皆对象

Python 里就连 `int`、`str`、函数、类、模块都是对象。

```python
# 函数是一等公民
def greet(name):
    return f"hi {name}"

say_hello = greet           # 赋值给变量
funcs = [greet, len, str]   # 放列表里
d = {"fn": greet}           # 放字典里

def call_twice(f, x):
    return f(f(x))

print(call_twice(str.upper, "hi"))  # "HI"
```

```cpp
// C++ 函数不是一等公民（但函数指针/lambda 可以）
auto greet = [](auto name) { return "hi "s + name; };
std::vector<std::function<std::string(std::string)>> funcs = {greet};
```

**Python 对象三要素**：`id()`（身份）、`type()`（类型）、`value`（值）。

## Duck Typing

> "如果它走路像鸭子、叫起来像鸭子，那它就是鸭子。"

```python
class Duck:
    def quack(self): print("嘎嘎")

class Person:
    def quack(self): print("学鸭子叫")

def make_it_quack(thing):
    thing.quack()  # 不关心类型，只关心有没有 quack 方法

make_it_quack(Duck())    # 嘎嘎
make_it_quack(Person())  # 学鸭子叫
```

```go
// Go 的 interface 也是 duck typing（结构型）
type Quacker interface {
    Quack()
}
```

```cpp
// C++ 用模板实现 duck typing（编译期）
template <typename T>
void make_it_quack(T& thing) { thing.quack(); }
```

## 与 C++/Go 的思维对比

| 概念 | Python | C++ | Go |
|---|---|---|---|
| 类型绑定 | 运行时 | 编译时 | 编译时 |
| 变量本质 | 名字→对象引用 | 名字→内存位置 | 名字→内存位置（值语义） |
| 函数 | 一等公民 | 函数指针/lambda | 一等公民 |
| 多态 | duck typing（运行时） | 虚函数（运行时）or 模板（编译时） | interface（结构型） |
| 内存管理 | GC + 引用计数 | 手动/RAII | GC（并发标记清除） |

## 总结

- Python 的"一切皆对象"是理解一切的基础——包括函数传递、类机制、元类
- Duck typing 让 Python 代码更灵活，但**没有类型保障**——这就是为什么后来加了 type hints
- 变量是"标签"贴在对象上，不是"盒子"存值——这一点和 C++/Go 完全不同
