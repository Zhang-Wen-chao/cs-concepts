"""函数深入：默认参数陷阱、*args/**kwargs、闭包、LEGB"""

# === 默认参数陷阱 ===
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target


def bad_append_to(element, target=[]):
    target.append(element)
    return target


# === *args / **kwargs ===
def sum_all(*args):
    return sum(args)


def create_url(**kwargs):
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    return "&".join(parts)


# === closure 和 LEGB ===
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter


def make_multiplier(x):
    def multiply(y):
        return x * y
    return multiply


# === 闭包陷阱 ===
def make_functions_bad():
    funcs = []
    for i in range(3):
        funcs.append(lambda: i)
    return funcs


def make_functions_good():
    funcs = []
    for i in range(3):
        funcs.append(lambda x=i: x)
    return funcs
