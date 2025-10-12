# Memory Management - 内存管理

> 程序如何使用内存？如何分配和释放？如何避免内存泄漏？

## 🎯 什么是内存管理？

**内存管理**是程序在运行时如何使用和管理计算机内存的过程，包括：
- 分配内存空间
- 使用内存存储数据
- 释放不再使用的内存

**为什么重要？**
- 内存是有限资源
- 不当使用导致崩溃、泄漏、性能问题
- 影响程序的可靠性和效率

---

## 🏗️ 内存的基本结构

程序的内存通常分为几个区域：

```
+-------------------+  高地址
|       栈          |  ← 局部变量、函数调用
|       ↓           |
|                   |
|       ↑           |
|       堆          |  ← 动态分配的内存
+-------------------+
|   BSS段(未初始化)  |  ← 未初始化的全局变量
+-------------------+
|   数据段(已初始化) |  ← 已初始化的全局变量
+-------------------+
|     代码段         |  ← 程序代码
+-------------------+  低地址
```

---

## 📦 栈 (Stack)

### 特点
- **自动管理** - 编译器自动分配和释放
- **快速** - 连续的内存，分配/释放只需移动指针
- **有限大小** - 通常几MB（栈溢出 Stack Overflow）
- **LIFO** - 后进先出

### 存储内容
- 局部变量
- 函数参数
- 返回地址

### 例子

```c
// C语言示例
void function() {
    int x = 10;      // 在栈上分配
    int arr[100];    // 在栈上分配数组
    // 函数返回时，自动释放
}
```

```python
# Python示例（概念相同）
def calculate():
    x = 10           # 引用在栈上
    y = 20           # 引用在栈上
    result = x + y
    return result    # 返回后局部变量被清理
```

### 栈的生命周期

```
调用函数 → 压入栈帧 → 执行代码 → 弹出栈帧 → 内存自动释放
```

### 栈溢出示例

```python
def infinite_recursion(n):
    return infinite_recursion(n + 1)  # 无限递归导致栈溢出

infinite_recursion(0)  # ❌ RecursionError: maximum recursion depth exceeded
```

---

## 🌊 堆 (Heap)

### 特点
- **手动/自动管理** - 需要显式分配，释放取决于语言
- **较慢** - 需要查找合适的空闲块
- **大容量** - 通常可用几GB
- **灵活** - 大小可变，生命周期长

### 存储内容
- 动态分配的对象
- 大型数据结构
- 生命周期超出函数范围的数据

### 例子

```python
# Python - 对象在堆上
class Person:
    def __init__(self, name):
        self.name = name

person = Person("Alice")  # Person对象在堆上分配
                          # person变量（引用）在栈上

# 大型列表在堆上
large_list = [0] * 1000000
```

```c
// C语言 - 手动管理堆内存
#include <stdlib.h>

void example() {
    // 在堆上分配内存
    int* numbers = (int*)malloc(100 * sizeof(int));

    // 使用内存
    numbers[0] = 42;

    // 必须手动释放
    free(numbers);  // ⚠️ 忘记这一步会内存泄漏！
}
```

---

## 🔄 内存分配方式

### 1. 静态分配
编译时确定大小和位置

```c
int global_var = 42;  // 编译时分配在数据段

void function() {
    static int counter = 0;  // 第一次调用时初始化，程序结束时释放
    counter++;
}
```

### 2. 自动分配（栈分配）
进入作用域时分配，离开时释放

```c
void function() {
    int x = 10;  // 进入函数时分配
    {
        int y = 20;  // 进入块时分配
    }  // y自动释放
}  // x自动释放
```

### 3. 动态分配（堆分配）
运行时按需分配

```c
// C
int* ptr = (int*)malloc(sizeof(int));  // 分配
*ptr = 42;
free(ptr);  // 释放

// C++
int* ptr = new int(42);  // 分配
delete ptr;  // 释放

// 数组
int* arr = new int[100];
delete[] arr;
```

---

## 🗑️ 内存释放策略

### 1. 手动管理
程序员负责分配和释放

```c
// C/C++
void example() {
    char* buffer = (char*)malloc(1024);
    if (buffer == NULL) {
        // 处理分配失败
        return;
    }

    // 使用buffer...

    free(buffer);  // 必须手动释放
}
```

**常见问题**：
```c
// ❌ 内存泄漏
char* buffer = malloc(1024);
buffer = malloc(2048);  // 前一块内存泄漏了！

// ❌ 悬空指针
char* ptr = malloc(100);
free(ptr);
*ptr = 'A';  // 危险！使用已释放的内存

// ❌ 重复释放
char* ptr = malloc(100);
free(ptr);
free(ptr);  // 崩溃！
```

### 2. 垃圾回收 (Garbage Collection, GC)
自动检测和释放不再使用的内存

#### 引用计数 (Reference Counting)

```python
# Python使用引用计数
x = [1, 2, 3]  # 引用计数 = 1
y = x          # 引用计数 = 2
del x          # 引用计数 = 1
del y          # 引用计数 = 0 → 自动释放
```

**问题**：循环引用

```python
class Node:
    def __init__(self):
        self.next = None

a = Node()
b = Node()
a.next = b
b.next = a  # 循环引用！

del a
del b  # 引用计数都不为0，但已无法访问 → 内存泄漏
```

#### 标记-清除 (Mark and Sweep)

```
1. 标记阶段：从根对象开始，标记所有可达对象
2. 清除阶段：回收未标记的对象
```

```python
# Python的垃圾回收器处理循环引用
import gc

class Node:
    def __init__(self):
        self.next = None

a = Node()
b = Node()
a.next = b
b.next = a

del a
del b
gc.collect()  # 强制垃圾回收，清理循环引用
```

#### 分代回收 (Generational GC)

```
假设：
- 大多数对象存活时间短
- 老对象很少引用新对象

策略：
- 新生代：频繁检查
- 老年代：较少检查
```

### 3. 所有权系统 (Ownership)
编译时保证内存安全（Rust）

```rust
// Rust - 所有权规则
fn main() {
    let s1 = String::from("hello");  // s1拥有字符串
    let s2 = s1;  // 所有权转移给s2，s1不再有效

    // println!("{}", s1);  // ❌ 编译错误！s1已失效

    println!("{}", s2);  // ✅

}  // s2离开作用域，自动释放内存

// 借用规则
fn calculate_length(s: &String) -> usize {
    s.len()  // 借用s，不获取所有权
}  // s不会被释放，因为没有所有权

let s = String::from("hello");
let len = calculate_length(&s);  // 传递引用
println!("{}", s);  // ✅ s仍然有效
```

---

## 🐛 常见内存问题

### 1. 内存泄漏 (Memory Leak)
分配的内存没有释放

### 2. 悬空指针 (Dangling Pointer)
指向已释放内存的指针

### 3. 缓冲区溢出 (Buffer Overflow)
写入超出分配的内存边界

### 4. 双重释放 (Double Free)
释放同一块内存两次

---

## 💡 最佳实践

1. **RAII** - 资源获取即初始化
2. **使用智能指针** - 自动管理生命周期
3. **对象池** - 复用对象减少分配
4. **及时释放** - 不再使用立即释放

---

## 🔗 相关概念

- [编程范式](programming-paradigms.md) - 范式影响内存管理方式
- [类型系统](type-systems.md) - 类型决定内存布局
- [并发编程](concurrency-parallelism.md) - 多线程的内存可见性
- [操作系统](../../systems/operating-systems/) - 虚拟内存、分页机制
