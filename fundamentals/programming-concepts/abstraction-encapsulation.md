# Abstraction and Encapsulation - 抽象与封装

> 如何隐藏复杂性？如何设计清晰的接口？

## 🎯 核心概念

### 抽象 (Abstraction)
**提取关键特征，忽略无关细节**

```
汽车抽象：
关注：加速、刹车、转向
忽略：发动机内部、变速箱细节
```

### 封装 (Encapsulation)
**隐藏内部实现，只暴露必要接口**

```
电视遥控器：
暴露：音量、频道、开关按钮
隐藏：电路板、信号处理、芯片
```

---

## 🔍 抽象 vs 封装

| 特性 | 抽象 (Abstraction) | 封装 (Encapsulation) |
|-----|-------------------|---------------------|
| **关注点** | 做什么 (What) | 怎么做 (How) |
| **目的** | 简化复杂性 | 隐藏实现 |
| **层次** | 设计层面 | 实现层面 |
| **例子** | 接口定义 | 私有变量 |

**关系**：抽象是思想，封装是实现抽象的手段

---

## 📦 封装的实现

### 1. 访问控制

```python
class BankAccount:
    def __init__(self):
        self.__balance = 0  # 私有属性（Python用__表示）

    def deposit(self, amount):  # 公开方法
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def get_balance(self):  # 公开方法
        return self.__balance

    def __validate(self, amount):  # 私有方法
        return amount > 0

# 使用
account = BankAccount()
account.deposit(100)  # ✅ 通过公开接口
print(account.get_balance())  # ✅

# account.__balance = 999999  # ❌ 不能直接访问（Python会改名）
```

```java
// Java - 更严格的访问控制
public class BankAccount {
    private double balance;  // 私有

    public void deposit(double amount) {  // 公开
        if (amount > 0) {
            balance += amount;
        }
    }

    public double getBalance() {  // 公开
        return balance;
    }

    private boolean validate(double amount) {  // 私有
        return amount > 0;
    }
}
```

### 2. 属性 (Property)

```python
class Temperature:
    def __init__(self):
        self._celsius = 0  # 内部存储

    @property
    def celsius(self):  # getter
        return self._celsius

    @celsius.setter
    def celsius(self, value):  # setter
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value

    @property
    def fahrenheit(self):  # 计算属性
        return self._celsius * 9/5 + 32

# 使用
temp = Temperature()
temp.celsius = 25  # 看起来像直接赋值，实际调用setter
print(temp.fahrenheit)  # 自动计算
# temp.celsius = -300  # ❌ 抛出异常
```

### 3. 模块化

```python
# math_utils.py - 封装数学工具
def _internal_helper(x):  # 内部函数（约定用_开头）
    return x * 2

def public_function(x):  # 公开函数
    return _internal_helper(x) + 1

# 使用时
from math_utils import public_function
# 只能看到公开的函数
```

---

## 🎨 抽象的实现

### 1. 抽象类

```python
from abc import ABC, abstractmethod

class Shape(ABC):  # 抽象类
    @abstractmethod
    def area(self):  # 抽象方法：只定义接口，不实现
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):  # 必须实现
        return 3.14 * self.radius ** 2

    def perimeter(self):  # 必须实现
        return 2 * 3.14 * self.radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # ❌ 不能实例化抽象类
circle = Circle(5)  # ✅
print(circle.area())
```

### 2. 接口

```python
# Python的接口（Protocol）
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Printable(Protocol):
    def print(self) -> None:
        ...

# 任何实现了draw方法的类都满足Drawable接口
class Circle:
    def draw(self) -> str:
        return "绘制圆形"

class Square:
    def draw(self) -> str:
        return "绘制方形"

def render(shape: Drawable):  # 接受任何Drawable
    print(shape.draw())

render(Circle())  # ✅
render(Square())  # ✅
```

```java
// Java的接口更明确
interface Drawable {
    void draw();
}

interface Resizable {
    void resize(int width, int height);
}

// 类可以实现多个接口
class Rectangle implements Drawable, Resizable {
    public void draw() {
        System.out.println("绘制矩形");
    }

    public void resize(int w, int h) {
        this.width = w;
        this.height = h;
    }
}
```

### 3. 鸭子类型（Duck Typing）

```python
# Python的动态类型抽象
class Duck:
    def quack(self):
        return "嘎嘎！"

class Person:
    def quack(self):
        return "我在模仿鸭子！"

class Dog:
    def bark(self):
        return "汪汪！"

def make_it_quack(thing):
    # 只关心有没有quack方法，不关心类型
    return thing.quack()

make_it_quack(Duck())    # ✅
make_it_quack(Person())  # ✅
# make_it_quack(Dog())   # ❌ 没有quack方法

# "如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子"
```

---

## 🏗️ 抽象层次

### 多层抽象

```python
# 底层：具体实现
class FileStorage:
    def save(self, data, filename):
        with open(filename, 'w') as f:
            f.write(data)

class DatabaseStorage:
    def save(self, data, table):
        # 保存到数据库
        pass

# 中层：抽象接口
class Storage(ABC):
    @abstractmethod
    def save(self, data, location):
        pass

# 高层：业务逻辑
class UserManager:
    def __init__(self, storage: Storage):
        self.storage = storage  # 依赖抽象，不依赖具体

    def save_user(self, user):
        self.storage.save(user.to_json(), user.id)

# 使用
file_storage = FileStorage()
user_manager = UserManager(file_storage)  # 可以随时换成数据库存储
```

### 依赖倒置原则

```
❌ 错误：高层依赖低层
UserManager → FileStorage（具体实现）

✅ 正确：都依赖抽象
UserManager → Storage（抽象接口）← FileStorage
                       ← DatabaseStorage
```

---

## 💡 设计原则

### 1. 单一职责原则 (SRP)
**一个类只做一件事**

```python
# ❌ 职责太多
class User:
    def save_to_db(self):
        pass
    def send_email(self):
        pass
    def generate_report(self):
        pass

# ✅ 职责分离
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class UserRepository:
    def save(self, user):
        pass

class EmailService:
    def send(self, to, message):
        pass

class ReportGenerator:
    def generate(self, user):
        pass
```

### 2. 开闭原则 (OCP)
**对扩展开放，对修改封闭**

```python
# ❌ 需要修改原代码
class PaymentProcessor:
    def process(self, method, amount):
        if method == "credit_card":
            # 处理信用卡
            pass
        elif method == "paypal":
            # 处理PayPal
            pass
        # 添加新支付方式需要修改这里 ❌

# ✅ 通过继承扩展
class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount):
        pass

class CreditCard(PaymentMethod):
    def process(self, amount):
        print(f"信用卡支付 {amount}")

class PayPal(PaymentMethod):
    def process(self, amount):
        print(f"PayPal支付 {amount}")

class Bitcoin(PaymentMethod):  # 新增支付方式，无需修改原代码 ✅
    def process(self, amount):
        print(f"比特币支付 {amount}")

class PaymentProcessor:
    def process(self, method: PaymentMethod, amount):
        method.process(amount)
```

### 3. 里氏替换原则 (LSP)
**子类可以替换父类**

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)

    # 可以安全替换Rectangle使用
    def area(self):
        return self.width * self.width

def print_area(rect: Rectangle):
    print(f"面积: {rect.area()}")

print_area(Rectangle(4, 5))  # ✅
print_area(Square(4))        # ✅ 可以替换
```

### 4. 接口隔离原则 (ISP)
**客户端不应依赖它不需要的接口**

```python
# ❌ 接口太大
class Worker(ABC):
    @abstractmethod
    def work(self): pass

    @abstractmethod
    def eat(self): pass

class Robot(Worker):  # 机器人不需要eat ❌
    def work(self): pass
    def eat(self): pass  # 不需要但必须实现

# ✅ 接口分离
class Workable(ABC):
    @abstractmethod
    def work(self): pass

class Eatable(ABC):
    @abstractmethod
    def eat(self): pass

class Human(Workable, Eatable):
    def work(self): pass
    def eat(self): pass

class Robot(Workable):  # 只实现需要的 ✅
    def work(self): pass
```

### 5. 依赖倒置原则 (DIP)
**依赖抽象而非具体**

```python
# ❌ 依赖具体实现
class EmailNotifier:
    def send(self, message):
        print(f"发送邮件: {message}")

class UserService:
    def __init__(self):
        self.notifier = EmailNotifier()  # 绑死了

    def register(self, user):
        # 注册用户...
        self.notifier.send("欢迎！")

# ✅ 依赖抽象
class Notifier(ABC):
    @abstractmethod
    def send(self, message): pass

class EmailNotifier(Notifier):
    def send(self, message):
        print(f"发送邮件: {message}")

class SMSNotifier(Notifier):
    def send(self, message):
        print(f"发送短信: {message}")

class UserService:
    def __init__(self, notifier: Notifier):
        self.notifier = notifier  # 依赖抽象，灵活

    def register(self, user):
        # 注册用户...
        self.notifier.send("欢迎！")

# 可以随时切换
service = UserService(EmailNotifier())
service = UserService(SMSNotifier())
```

---

## 🎯 实际应用

### 1. API设计

```python
# 设计清晰的API
class Cache:
    """简单的缓存接口"""

    def get(self, key: str) -> any:
        """获取缓存值"""
        pass

    def set(self, key: str, value: any, ttl: int = None):
        """设置缓存，可选过期时间"""
        pass

    def delete(self, key: str):
        """删除缓存"""
        pass

    def clear(self):
        """清空所有缓存"""
        pass

# 使用者不需要知道内部实现（内存？Redis？）
```

### 2. 框架设计

```python
# Web框架的抽象
class View(ABC):
    @abstractmethod
    def get(self, request):
        """处理GET请求"""
        pass

    @abstractmethod
    def post(self, request):
        """处理POST请求"""
        pass

class UserView(View):
    def get(self, request):
        return {"users": [...]}

    def post(self, request):
        # 创建用户
        return {"status": "created"}

# 框架只需要知道View接口，不关心具体实现
```

### 3. 数据访问层

```python
class Repository(ABC):
    @abstractmethod
    def find_by_id(self, id): pass

    @abstractmethod
    def save(self, entity): pass

    @abstractmethod
    def delete(self, entity): pass

class SQLRepository(Repository):
    def find_by_id(self, id):
        # SQL查询
        pass

class MongoRepository(Repository):
    def find_by_id(self, id):
        # MongoDB查询
        pass

# 业务逻辑不关心用的是SQL还是MongoDB
```

---

## 🔗 相关概念

- [编程范式](programming-paradigms.md) - OOP强调封装和抽象
- [类型系统](type-systems.md) - 接口是类型抽象
- [设计模式](../../software-engineering/design-patterns/) - 抽象和封装的具体应用
- [软件架构](../../software-engineering/software-architecture/) - 系统级的抽象

---

## 📚 深入学习

- **书籍**：《设计模式》、《Clean Code》、《代码整洁之道》
- **原则**：SOLID原则
- **实践**：重构、接口设计、API设计

---

**记住**：
1. 抽象关注"做什么"，封装关注"怎么做"
2. 好的抽象让代码更简单，坏的抽象让代码更复杂
3. 封装保护数据完整性
4. 依赖抽象而非具体
5. 接口应该小而精确
