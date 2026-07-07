"""异常处理：自定义异常、异常链、contextlib.suppress"""

# === 自定义异常 ===
class ConfigError(Exception):
    pass

class ConfigNotFoundError(ConfigError):
    def __init__(self, path):
        self.path = path
        super().__init__(f"config not found: {path}")

class ConfigParseError(ConfigError):
    def __init__(self, path, detail):
        self.path = path
        self.detail = detail
        super().__init__(f"parse error in {path}: {detail}")


# === 异常链 ===
def load_config(path):
    try:
        with open(path) as f:
            return parse_config(f.read())
    except FileNotFoundError as e:
        raise ConfigNotFoundError(path) from e


def parse_config(content):
    if "{" not in content:
        raise ConfigParseError("inline", "missing braces")
    return {"parsed": True}


# === try/except/else/finally ===
def divide_with_else(a, b):
    results = {"success": False, "result": None}
    try:
        results["result"] = a / b
    except ZeroDivisionError:
        results["result"] = float("inf")
    else:
        results["success"] = True
    finally:
        results["called"] = True
    return results


# === contextlib.suppress ===
from contextlib import suppress

def clean_filenames(names):
    result = []
    for name in names:
        with suppress(ValueError):
            result.append(int(name))
    return result
