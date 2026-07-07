# 集成测试策略

## 测试层级

| 层级 | Python 工具 | 说明 |
|------|-------------|------|
| 单元测试 | pytest | 测试各模块独立行为 |
| 集成测试 | pytest + httpx | 测试 API 端点 |
| 端到端 | subprocess + golden files | 测试 CLI 命令行 |
| 性能测试 | pytest-benchmark | 压测解析/聚合性能 |

## CLI 测试

```python
import subprocess
import json

def test_cli_json_output():
    result = subprocess.run(
        ["python", "-m", "log_analyzer", "--mode=cli",
         "--input", "testdata/access.log", "--format=json"],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    assert "total_requests" in data
    assert data["total_requests"] > 0
```

## API 测试

```python
from httpx import AsyncClient
from log_analyzer.server import app

async def test_ingest_and_query():
    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/v1/ingest", json={"line": '...'})
        assert resp.status_code == 200

        resp = await client.get("/v1/query")
        data = resp.json()
        assert "stats" in data
```

## Golden Files

将预期输出存为 `.golden` 文件，测试时对比：

```python
def test_cli_output_matches_golden(tmp_path):
    output_path = tmp_path / "output.json"
    subprocess.run([... "--output", str(output_path)])
    golden = (TESTDATA / "expected_output.json").read_text()
    assert output_path.read_text() == golden
```
