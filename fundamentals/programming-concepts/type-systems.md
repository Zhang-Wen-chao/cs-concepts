# Type Systems - 类型系统

> 类型是什么？为什么需要类型？类型系统如何帮助我们写出更好的代码？

## 🎯 什么是类型？

**类型**是对数据的分类，定义了：
- 数据可以存储什么值
- 可以进行哪些操作
- 如何在内存中表示

```python
x = 42        # 整数类型：可以加减乘除
name = "Alice"  # 字符串类型：可以拼接、切片
items = [1, 2]  # 列表类型：可以添加、删除元素
```

**类型系统**是编程语言用来检查和强制类型规则的机制。

---

## 📊 基本类型分类

### 1. 原始类型 (Primitive Types)
语言内置的基础类型

```python
# Python
integer = 42           # 整数
floating = 3.14        # 浮点数
boolean = True         # 布尔值
string = "hello"       # 字符串
none_val = None        # 空值
```

```javascript
// JavaScript
let num = 42;          // number
let str = "hello";     // string
let bool = true;       // boolean
let nothing = null;    // null
let undef = undefined; // undefined
```

### 2. 复合类型 (Composite Types)
由多个值组合而成

```python
# 列表/数组
numbers = [1, 2, 3, 4]

# 字典/映射
person = {"name": "Alice", "age": 30}

# 元组
point = (10, 20)

# 集合
unique_numbers = {1, 2, 3}
```

### 3. 自定义类型 (User-Defined Types)
程序员定义的新类型

```python
# 类
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 枚举
from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
```

---

## 🔍 静态类型 vs 动态类型

### 静态类型 (Static Typing)
**编译时检查类型**

```java
// Java - 静态类型
int age = 30;
age = "thirty";  // ❌ 编译错误！类型不匹配
```

**特点**：
- ✅ 编译时捕获类型错误
- ✅ 更好的IDE支持（自动完成、重构）
- ✅ 性能优化空间大
- ❌ 代码冗长
- ❌ 灵活性较低

**语言示例**：Java, C++, C#, TypeScript, Rust, Go

### 动态类型 (Dynamic Typing)
**运行时检查类型**

```python
# Python - 动态类型
age = 30
age = "thirty"  # ✅ 运行时才知道类型变了
print(age + 10)  # ❌ 运行时报错！
```

**特点**：
- ✅ 代码简洁
- ✅ 灵活性高
- ✅ 快速原型开发
- ❌ 运行时才发现类型错误
- ❌ 重构困难

**语言示例**：Python, JavaScript, Ruby, PHP

---

## 💪 强类型 vs 弱类型

### 强类型 (Strong Typing)
**严格的类型转换规则**

```python
# Python - 强类型
"3" + 5  # ❌ TypeError: 不能将字符串和整数相加
"3" + str(5)  # ✅ 必须显式转换
```

### 弱类型 (Weak Typing)
**宽松的类型转换规则**

```javascript
// JavaScript - 弱类型
"3" + 5   // ✅ "35" - 自动转为字符串拼接
"3" - 5   // ✅ -2 - 自动转为数字运算
```

**强弱类型是一个光谱，不是非黑即白：**

```
强类型 ←----------------------→ 弱类型
Haskell  Python  Java  C  JavaScript
```

---

## 🧩 类型推导 (Type Inference)

编译器/解释器自动推断变量类型

```typescript
// TypeScript
let x = 42;  // 推导为 number
let name = "Alice";  // 推导为 string

function add(a: number, b: number) {
    return a + b;  // 返回类型自动推导为 number
}
```

```rust
// Rust
let x = 42;  // 推导为 i32（32位整数）
let nums = vec![1, 2, 3];  // 推导为 Vec<i32>

fn double(x: i32) -> i32 {
    x * 2  // 返回类型已声明，但也可以推导
}
```

**优势**：
- 静态类型的安全性
- 动态类型的简洁性
- 最佳平衡

---

## 🎭 泛型与多态

### 泛型 (Generics)
编写适用于多种类型的代码

```python
# Python - 类型提示的泛型
from typing import List, TypeVar

T = TypeVar('T')

def first_element(items: List[T]) -> T:
    return items[0]

# 适用于任何类型的列表
numbers = first_element([1, 2, 3])  # 返回 int
names = first_element(["Alice", "Bob"])  # 返回 str
```

```java
// Java - 泛型类
class Box<T> {
    private T item;

    public void set(T item) {
        this.item = item;
    }

    public T get() {
        return item;
    }
}

Box<Integer> intBox = new Box<>();
Box<String> strBox = new Box<>();
```

### 多态 (Polymorphism)

#### 1. 参数多态 (Parametric Polymorphism)
就是泛型

#### 2. 子类型多态 (Subtype Polymorphism)
面向对象的多态

```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "汪汪"

class Cat(Animal):
    def speak(self):
        return "喵喵"

def make_sound(animal: Animal):  # 接受任何Animal子类
    print(animal.speak())

make_sound(Dog())  # "汪汪"
make_sound(Cat())  # "喵喵"
```

#### 3. Ad-hoc多态 (Ad-hoc Polymorphism)
函数重载

```cpp
// C++ - 函数重载
int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

string add(string a, string b) {
    return a + b;
}
```

---

## 🛡️ 类型安全

### 什么是类型安全？
程序不会因为类型错误而产生未定义行为

```python
# 类型不安全的例子
def divide(a, b):
    return a / b

divide(10, 0)  # ❌ 运行时错误：除以零
divide("10", "2")  # ❌ 运行时错误：字符串不能除
```

```python
# 类型安全的改进
from typing import Union

def divide(a: float, b: float) -> Union[float, None]:
    if b == 0:
        return None
    return a / b

result = divide(10.0, 2.0)
if result is not None:
    print(result)
```

### Option/Maybe类型
处理可能不存在的值

```rust
// Rust - Option类型
fn divide(a: i32, b: i32) -> Option<i32> {
    if b == 0 {
        None
    } else {
        Some(a / b)
    }
}

match divide(10, 2) {
    Some(result) => println!("结果: {}", result),
    None => println!("除数不能为零"),
}
```

---

## 🔗 类型系统的实际应用

### 1. 类型注解/类型提示

```python
# Python 3.5+ 类型提示
def greet(name: str) -> str:
    return f"Hello, {name}!"

from typing import List, Dict, Optional

def process_users(users: List[Dict[str, str]]) -> Optional[str]:
    if not users:
        return None
    return users[0]["name"]
```

### 2. 类型检查工具

```bash
# mypy - Python类型检查器
$ mypy my_program.py

# TypeScript编译器
$ tsc --strict my_program.ts
```

### 3. 接口与协议

```python
# Python - 协议（结构化子类型）
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "绘制圆形"

class Square:
    def draw(self) -> str:
        return "绘制方形"

def render(shape: Drawable):  # 只要有draw方法就行
    print(shape.draw())
```

---

## 📐 高级类型概念

### 1. 联合类型 (Union Types)

```python
from typing import Union

def process(value: Union[int, str]) -> str:
    if isinstance(value, int):
        return f"数字: {value}"
    else:
        return f"字符串: {value}"
```

### 2. 交叉类型 (Intersection Types)

```typescript
// TypeScript
type Named = { name: string };
type Aged = { age: number };

type Person = Named & Aged;  // 必须同时有name和age

const person: Person = {
    name: "Alice",
    age: 30
};
```

### 3. 字面量类型 (Literal Types)

```typescript
// TypeScript
type Direction = "north" | "south" | "east" | "west";

function move(direction: Direction) {
    // direction只能是这四个值之一
}

move("north");  // ✅
move("up");     // ❌ 类型错误
```

### 4. 类型别名 (Type Aliases)

```python
from typing import List, Tuple

# 定义复杂类型的别名
Point = Tuple[float, float]
Path = List[Point]

def draw_line(path: Path) -> None:
    for point in path:
        print(f"点: {point}")
```

---

## 🎯 选择类型系统的考虑

| 需求 | 推荐 |
|-----|------|
| **大型项目** | 静态类型（早期发现错误） |
| **快速原型** | 动态类型（快速迭代） |
| **团队协作** | 静态类型（清晰接口） |
| **脚本任务** | 动态类型（灵活性） |
| **性能关键** | 静态类型（编译优化） |
| **API设计** | 静态类型（文档化） |

---

## 💡 实践建议

### 1. 渐进式类型系统
即使在动态语言中也使用类型提示

```python
# 逐步添加类型注解
def process_data(data):  # 开始
    ...

def process_data(data: dict):  # 添加基本类型
    ...

def process_data(data: Dict[str, Any]):  # 更精确
    ...

def process_data(data: Dict[str, Union[str, int]]):  # 最精确
    ...
```

### 2. 使用类型检查工具
- Python: mypy, pyright
- JavaScript: TypeScript, Flow
- PHP: Psalm, PHPStan

### 3. 文档化类型
即使语言不强制，也要在注释中说明

```python
def calculate_total(items):
    """
    计算总价

    Args:
        items: List[Dict[str, float]] - 商品列表
               每个商品: {"price": float, "quantity": int}

    Returns:
        float - 总价
    """
    return sum(item["price"] * item["quantity"] for item in items)
```

---

## 🔗 相关概念

- [编程范式](programming-paradigms.md) - 范式影响类型系统设计
- [内存管理](memory-management.md) - 类型影响内存布局
- [抽象与封装](abstraction-encapsulation.md) - 类型是抽象的工具
- [错误处理](error-handling.md) - 类型系统可以表达错误

---

## 📚 深入学习

- **TypeScript** - JavaScript的类型超集，学习现代类型系统
- **Rust** - 强大的类型系统和所有权概念
- **Haskell** - 学术级的类型系统
- **书籍**：《Types and Programming Languages》

---

**记住**：类型系统不是束缚，而是工具。好的类型系统能够：
1. 在编写代码时提供引导
2. 在修改代码时提供保护
3. 在阅读代码时提供文档

选择合适的类型系统，让它成为你的助手而非负担！
