# 测试：pytest 深入、fixture、parametrize、mock、覆盖率、TDD

> pytest 是 Python 事实上的测试标准。unittest 兼容但语法繁琐。

## pytest 基础

```python
# test_calc.py — 文件以 test_ 开头或 _test 结尾
def add(a, b):
    return a + b

def test_add_basic():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0

def test_add_float():
    assert add(0.1, 0.2) == pytest.approx(0.3)  # 浮点比较
```

```bash
# 运行
pytest                           # 自动发现测试
pytest -v                        # 详细输出
pytest -x                        # 首次失败后停止
pytest -k "float"                # 按名字过滤
pytest --pdb                     # 失败时进入调试器
```

## Fixture：测试依赖管理

```python
import pytest

@pytest.fixture
def db_connection():
    """为测试准备数据库连接"""
    conn = create_db_connection()
    yield conn                    # 测试期间使用 conn
    conn.close()                  # 测试后清理

@pytest.fixture(scope="session")
def config():
    """跨所有测试复用（只创建一次）"""
    return load_config()          # session 级别：整个测试会话共享

@pytest.fixture(scope="module")
def cache():
    """模块级别：一个模块内共享"""
    return Cache()

def test_query(db_connection):
    result = db_connection.query("SELECT 1")
    assert result == 1

def test_write(db_connection):
    db_connection.write("test")
    assert True
```

**fixture scope**：`function`（默认）→ `class` → `module` → `session`

## Parametrize：多组参数

```python
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add(a, b, expected):
    assert add(a, b) == expected

# 联合参数化
@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", [10, 20])
def test_cartesian(a, b):
    print(f"a={a}, b={b} — 组合了 {4} 种情况")
```

## Mock：隔离外部依赖

```python
import pytest
from unittest.mock import Mock, patch

# 1. Mock 对象
class TestAPI:
    def test_fetch_user(self):
        mock_response = Mock()
        mock_response.json.return_value = {"id": 1, "name": "Alice"}
        
        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_response
            result = fetch_user(1)
            assert result["name"] == "Alice"
    
    # 2. patch 装饰器
    @patch("mymodule.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = ConnectionError("timeout")
        with pytest.raises(ConnectionError):
            fetch_user(1)
    
    # 3. 监视方法调用
    def test_send_email(self):
        mail_service = Mock()
        process_order(mail_service, order_id=42)
        mail_service.send.assert_called_once_with(
            to="user@example.com",
            subject="订单确认 #42"
        )
```

## 覆盖率

```bash
# 安装
pip install pytest-cov

# 运行
pytest --cov=src tests/
pytest --cov=src --cov-report=html tests/  # 生成 HTML 报告
```

```python
# .coveragerc
[run]
source = src
omit = */tests/*,*/migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

## TDD 实践

```python
# 1️⃣ 先写测试
def test_parse_csv():
    result = parse_csv("a,b,c\n1,2,3")
    assert result == [{"a": "1", "b": "2", "c": "3"}]

# 2️⃣ 看到失败（RED）
# 3️⃣ 写最少代码让它过（GREEN）

def parse_csv(text):
    lines = text.strip().split("\n")
    headers = lines[0].split(",")
    return [dict(zip(headers, line.split(","))) for line in lines[1:]]

# 4️⃣ 重构（REFACTOR）
# 5️⃣ 重复

# 先写测试的好处：
# - 驱动设计（API 好不好用先试一遍）
# - 安全地重构
# - 文档即代码
```

## pytest vs unittest

```python
# unittest 写法（你不需要写这个，但能看懂）
import unittest

class TestCalc(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
    
    def tearDown(self):
        self.calc.close()

# pytest 写法（推荐）
def test_add(calculator):  # fixture 注入
    assert calculator.add(2, 3) == 5
```

## 总结

| 概念 | 用途 |
|---|---|
| `pytest` | 测试框架，自动发现 `test_*.py` |
| `@pytest.fixture` | 准备/清理测试环境 |
| `@pytest.mark.parametrize` | 多组测试参数 |
| `unittest.mock` | 隔离外部依赖 |
| `pytest-cov` | 覆盖率报告 |
| TDD | 先写测试再写代码（红-绿-重构） |

- pytest 比 unittest 简洁 3 倍——新项目永远用 pytest
- fixture 替代了 `setUp`/`tearDown`，scope 机制更灵活
- mock 测试边界：网络、DB、文件、时间
- 测试不是负担——测试是你敢重构的信心
