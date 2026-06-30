# 模块与包管理：pyproject.toml、pip vs poetry/uv、虚拟环境、发布 PyPI

> Python 的包管理史上最混乱——但 pyproject.toml 正在统一一切。

## pyproject.toml：现代 Python 项目的标准配置

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "awesome-tool"
version = "0.1.0"
description = "一个很酷的工具"
requires-python = ">=3.10"
dependencies = [
    "requests>=2.28",
    "rich>=13",
]

[project.optional-dependencies]
dev = [
    "pytest>=7",
    "black",
    "ruff",
]

[project.scripts]
awesome = "awesome_tool.cli:main"  # 安装后生成 CLI 命令

[tool.black]
line-length = 88

[tool.ruff]
line-length = 88
```

**PEP 621** 明确了 `[project]` 段，替代了 `setup.py`、`setup.cfg`、`requirements.txt` 的大乱斗。

## pip 与虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

# 安装包
pip install requests
pip install -r requirements.txt
pip install -e .                # 开发模式安装当前包

# 冻结依赖
pip freeze > requirements.txt

# 注意：pip freeze 会包含所有包（包括依赖的依赖）
# 更好的做法：用 pip-compile（来自 pip-tools）或 poetry export
```

**`.venv` 放哪**：通常放项目根目录，`.gitignore` 里加 `.venv/`。

## pip vs poetry vs uv

```bash
# Poetry（功能完整，生态成熟）
poetry new myproject          # 新建项目
poetry add requests           # 添加依赖并安装
poetry add --dev pytest       # 开发依赖
poetry shell                  # 进入虚拟环境
poetry build && poetry publish # 构建并发布

# uv（Rust 实现，极速，兼容 pip 生态）
uv pip install requests       # 兼容 pip 接口（快 10-100x）
uv venv                       # 创建虚拟环境
uv sync                       # 同步 pyproject.toml 依赖
uv add requests               # 添加依赖
uv tool run black --help      # 运行工具无需安装
```

| 对比 | pip | poetry | uv |
|---|---|---|---|
| 速度 | 慢 | 中等 | 极快 |
| 依赖解析 | 弱（可能冲突） | 强（锁定文件） | 强（锁文件） |
| CI 友好 | ✅ | 略慢 | ✅ ✅ |
| 学习曲线 | 0 | 中 | 低 |
| 推荐 | 基础项目 | 复杂项目 | 新项目首选 |

## 发布 PyPI

```bash
# 1. 构建
python -m build

# 2. 上传
# TestPyPI（先试）
python -m twine upload --repository testpypi dist/*

# 正式
python -m twine upload dist/*

# 或一行 poetry：
poetry publish --build
```

**需要**：PyPI 账号 + API token（放 `~/.pypirc` 或环境变量）。

## Import 路径问题

```python
# 项目结构
# myproject/
# ├── pyproject.toml
# └── src/
#     └── mypackage/
#         ├── __init__.py
#         └── module.py

# 开发时：
pip install -e .  # 符号链接安装，src/mypackage 可被 import
# 或设置 PYTHONPATH=src
```

**src layout** 推荐：把代码放 `src/` 下，避免直接跑测试时 import 路径混乱。

## 总结

| 阶段 | 推荐工具 |
|---|---|
| 快速实验 | `python -m venv .venv` + pip |
| 标准项目 | `poetry` 或 `uv` |
| 发布 PyPI | `poetry publish` 或 `build` + `twine` |
| CI/CD | `uv sync` 最快 |
| 遗留项目 | 保持原样，逐步迁移到 pyproject.toml |

- `pyproject.toml` 是未来的唯一标准
- 虚拟环境隔离依赖，是 Python 工程的基石
- uv 是目前的性能之王，考虑迁移
