# Import 系统：模块搜索路径、\_\_init\_\_.py、相对导入、循环导入

## 模块搜索路径

```python
import sys

# Python 按顺序在这些路径找模块
for p in sys.path:
    print(p)

# 典型顺序：
# 1. 当前脚本所在目录（或 cwd）
# 2. PYTHONPATH 环境变量
# 3. 标准库目录
# 4. site-packages（第三方包）
```

```python
# 运行时修改搜索路径（不推荐，但有前）：
sys.path.insert(0, "/my/custom/path")
```

## `__init__.py` 的作用

```python
# mypackage/__init__.py
# 1. 标记目录为 Python 包
# 2. 控制 `from mypackage import *` 的内容
__all__ = ["module_a", "helper"]

# 3. 集中导入，简化外部使用
from .module_a import useful_func
from .helper import util

# 外部调用时：
# from mypackage import useful_func  # 直接可用
```

**Python 3.3+**：namespace package 不需要 `__init__.py`（PEP 420），但常规包还是推荐加上。

## 相对导入 vs 绝对导入

```python
# 目录结构：
# mypackage/
# ├── __init__.py
# ├── module_a.py
# ├── subpackage/
# │   ├── __init__.py
# │   └── module_b.py

# module_b.py 中：

# 绝对导入（推荐）
from mypackage import module_a
from mypackage.subpackage import module_b

# 相对导入（只能在包内用，不能在 __main__ 中用）
from .. import module_a       # 父包
from . import module_b       # 同包
from ..subpackage import x   # 兄弟包
```

**相对导入的限制**：
- 不能在 `__main__` 模块（入口脚本）中使用
- `..` 不能超出包顶层
- 有歧义时用绝对导入更清晰

**最佳实践**：大项目全部用绝对导入，小项目内部可以用相对导入。

## 循环导入及处理

```python
# a.py
from b import B

class A:
    def __init__(self):
        self.b = B()

# b.py
from a import A

class B:
    def __init__(self):
        self.a = A()

# 导入 a.py 时：
# a.py 开始执行 → 遇到 from b import B → 跳到 b.py
# b.py 开始执行 → 遇到 from a import A → 跳到 a.py（此时 a.py 还没定义完 A）
# ❌ ImportError: cannot import name 'A' from partially initialized module
```

**解法**：

```python
# 1. 延迟导入（在函数/方法内导入）
# a.py
class A:
    def __init__(self):
        from b import B  # 运行时才导入
        self.b = B()

# 2. 重构：把公共依赖抽到第三个模块
# common.py
class Base:
    pass

# a.py、b.py 都 import common

# 3. 用类型检查解决 type hints 的循环导入
# a.py
from __future__ import annotations  # PEP 563
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from b import B  # 只在类型检查时导入
```

## Import 的执行机制

```python
# import mymodule 实际上做了三件事：
# 1. 在 sys.modules 中查找（缓存）
# 2. 如果没找到，找到模块文件并执行其全部代码
# 3. 在当前命名空间绑定名字

import sys

print("mymodule" in sys.modules)  # 检查是否已导入
del sys.modules["mymodule"]       # 强制重新导入
```

**注意**：模块代码只在**首次导入时执行一次**。后续导入直接从 `sys.modules` 取缓存。

## 总结

- `sys.path` 决定 Python 去哪找模块，`sys.modules` 缓存已导入的模块
- `__init__.py` 标记包并控制导入接口
- 绝对导入比相对导入稳定，相对导入只能在包内用
- 循环导入通常意味着设计问题，延迟导入是应急方案
- Python 的 import 是"跑一遍模块代码"——import 有副作用，模块是单例
