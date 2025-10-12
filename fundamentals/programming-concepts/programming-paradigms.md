# Programming Paradigms - 编程范式

> 不同的编程思维方式：如何组织代码、如何思考问题、如何构建解决方案

## 🎯 什么是编程范式？

编程范式是**编程的思维方式**，定义了：
- 如何组织代码结构
- 如何表达计算过程
- 如何管理状态和数据
- 如何抽象和建模问题

**关键理解**：范式不是语言特性，而是编程哲学。同一个语言可以支持多种范式。

---

## 🔄 命令式编程 (Imperative Programming)

### 核心思想
**明确告诉计算机"怎么做"** - 一步一步的指令序列

### 特点
- 显式的控制流（for、while、if）
- 可变状态和变量
- 语句按顺序执行
- 直接操作内存

### 例子：计算数组总和

```python
# 命令式风格
def sum_array(arr):
    total = 0                    # 创建可变变量
    for i in range(len(arr)):    # 显式循环
        total += arr[i]          # 修改状态
    return total
```

### 优势与劣势
✅ 直观易懂，符合直觉
✅ 性能可控，接近硬件
❌ 状态管理复杂
❌ 难以并行化

---

## 📋 声明式编程 (Declarative Programming)

### 核心思想
**告诉计算机"要什么"** - 描述结果，而非过程

### 特点
- 隐藏控制流细节
- 关注"是什么"而非"怎么做"
- 更高层次的抽象
- 系统决定执行方式

### 例子：相同的求和任务

```python
# 声明式风格
def sum_array(arr):
    return sum(arr)  # 声明要什么：数组的和

# SQL也是声明式
SELECT SUM(price) FROM products WHERE category = 'books'
# 描述结果，不管数据库如何执行
```

### 优势与劣势
✅ 代码简洁清晰
✅ 更容易推理和优化
✅ 适合并行化
❌ 性能可能不透明
❌ 调试可能困难

---

## 🧱 面向对象编程 (Object-Oriented Programming, OOP)

### 核心思想
**用对象建模现实世界** - 数据和行为封装在一起

### 四大支柱

#### 1. 封装 (Encapsulation)
隐藏内部实现，只暴露必要接口

```python
class BankAccount:
    def __init__(self):
        self.__balance = 0  # 私有属性

    def deposit(self, amount):  # 公开接口
        if amount > 0:
            self.__balance += amount
```

#### 2. 继承 (Inheritance)
子类继承父类的属性和方法

```python
class Animal:
    def breathe(self):
        return "呼吸中..."

class Dog(Animal):  # 继承Animal
    def bark(self):
        return "汪汪！"
```

#### 3. 多态 (Polymorphism)
同一接口，不同实现

```python
class Shape:
    def area(self): pass

class Circle(Shape):
    def area(self):
        return 3.14 * self.radius ** 2

class Square(Shape):
    def area(self):
        return self.side ** 2

# 多态使用
shapes = [Circle(5), Square(4)]
for shape in shapes:
    print(shape.area())  # 同一接口，不同行为
```

#### 4. 抽象 (Abstraction)
提取关键特征，忽略细节

```python
from abc import ABC, abstractmethod

class Database(ABC):  # 抽象类
    @abstractmethod
    def connect(self): pass

    @abstractmethod
    def query(self, sql): pass
```

### 优势与劣势
✅ 易于理解和维护
✅ 代码复用性强
✅ 模拟现实世界
❌ 可能过度设计
❌ 继承层次过深会复杂

---

## 🔢 函数式编程 (Functional Programming, FP)

### 核心思想
**计算是函数求值** - 避免状态变化和可变数据

### 核心概念

#### 1. 纯函数 (Pure Functions)
相同输入 → 相同输出，无副作用

```python
# 纯函数
def add(a, b):
    return a + b  # 只依赖输入，不修改外部状态

# 非纯函数
total = 0
def add_to_total(x):
    global total
    total += x  # 修改外部状态，有副作用
```

#### 2. 不可变性 (Immutability)
数据一旦创建就不能修改

```python
# 函数式风格：创建新列表而非修改
def add_element(lst, elem):
    return lst + [elem]  # 返回新列表

# 命令式风格：直接修改
def add_element_imperative(lst, elem):
    lst.append(elem)  # 修改原列表
```

#### 3. 高阶函数 (Higher-Order Functions)
函数可以作为参数或返回值

```python
# map: 对每个元素应用函数
numbers = [1, 2, 3, 4]
squares = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16]

# filter: 筛选满足条件的元素
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

# reduce: 累积计算
from functools import reduce
total = reduce(lambda acc, x: acc + x, numbers)  # 10
```

#### 4. 函数组合 (Function Composition)
组合小函数构建复杂功能

```python
def double(x): return x * 2
def add_one(x): return x + 1

# 函数组合
def double_then_add_one(x):
    return add_one(double(x))

result = double_then_add_one(5)  # (5 * 2) + 1 = 11
```

### 优势与劣势
✅ 易于测试和推理
✅ 天然支持并行
✅ 避免状态bug
❌ 学习曲线陡峭
❌ 某些场景性能较低

---

## 🔀 过程式编程 (Procedural Programming)

### 核心思想
**用过程（函数）组织代码** - 命令式编程的结构化版本

### 特点
- 代码分解为可复用的过程/函数
- 自顶向下的设计方法
- 强调算法和数据结构分离

```python
# 过程式风格：分解为多个函数
def read_file(filename):
    with open(filename) as f:
        return f.read()

def process_data(data):
    return data.strip().upper()

def write_file(filename, data):
    with open(filename, 'w') as f:
        f.write(data)

def main():
    data = read_file('input.txt')
    processed = process_data(data)
    write_file('output.txt', processed)
```

### 优势与劣势
✅ 结构清晰，易于理解
✅ 代码复用性好
❌ 大型项目难以维护
❌ 数据和行为分离

---

## 🔄 范式对比

### 同一问题的不同范式实现

**问题**：从列表中筛选偶数并求和

```python
# 1. 命令式
def sum_evens_imperative(numbers):
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total

# 2. 函数式
def sum_evens_functional(numbers):
    return sum(filter(lambda x: x % 2 == 0, numbers))

# 3. 面向对象
class NumberProcessor:
    def __init__(self, numbers):
        self.numbers = numbers

    def sum_evens(self):
        return sum(n for n in self.numbers if n % 2 == 0)

# 使用
numbers = [1, 2, 3, 4, 5, 6]
print(sum_evens_imperative(numbers))  # 12
print(sum_evens_functional(numbers))  # 12
print(NumberProcessor(numbers).sum_evens())  # 12
```

---

## 🎯 选择范式的考虑因素

| 因素 | 推荐范式 |
|------|---------|
| **问题领域** | 业务模型 → OOP，数据转换 → FP |
| **团队熟悉度** | 团队最熟悉的范式 |
| **性能要求** | 命令式/过程式更可控 |
| **并发需求** | 函数式天然支持并行 |
| **可维护性** | OOP适合大型项目 |
| **快速原型** | 声明式/函数式更简洁 |

---

## 💡 实践建议

### 1. 多范式编程
现代语言通常支持多种范式，灵活选择：

```python
# Python支持多范式
class DataPipeline:  # OOP
    def __init__(self, data):
        self.data = data

    def process(self):  # 函数式风格的方法
        return (
            self.data
            .pipe(lambda x: x[x > 0])  # 链式调用
            .apply(lambda x: x ** 2)
            .sum()
        )
```

### 2. 根据场景选择
- **UI事件处理** → 面向对象
- **数据转换管道** → 函数式
- **系统脚本** → 过程式
- **查询处理** → 声明式

### 3. 学习多种范式
不同范式提供不同视角，丰富你的编程思维工具箱。

---

## 🔗 相关概念

- [类型系统](type-systems.md) - 不同范式对类型的处理方式
- [内存管理](memory-management.md) - 范式影响内存使用模式
- [并发编程](concurrency-parallelism.md) - 函数式范式的并发优势
- [设计模式](../../software-engineering/design-patterns/) - OOP中的常见模式
- [软件架构](../../software-engineering/software-architecture/) - 范式影响架构选择

---

## 📚 深入学习资源

- **书籍**：《设计模式》（OOP）、《SICP》（FP）
- **语言**：Java/C#（OOP）、Haskell/Clojure（FP）、C（过程式）
- **实践**：尝试用不同范式重写同一个项目

---

**记住**：没有最好的范式，只有最适合的范式。掌握多种范式，让你能够为每个问题选择最佳工具！
