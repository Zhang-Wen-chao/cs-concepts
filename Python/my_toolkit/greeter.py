import sys

def greet(name, language="en"):
    """根据语言向指定的人打招呼"""
    if language == "en":
        print(f"Hello, {name}!")
    elif language == "es":
        print(f"¡Hola, {name}!")
    elif language == "zh":
        print(f"你好, {name}！")
    else:
        print(f"Sorry, I don't speak '{language}'.")

# 这段代码是关键！
# 当 greeter.py 被当作脚本运行时 (__name__ == "__main__")，
# 它会检查命令行参数并调用 greet 函数。
if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        # 如果没有参数，就向 "World" 打招呼
        greet("World")
    elif len(args) == 1:
        # 如果有一个参数，就把它当作名字
        greet(args[0])
    elif len(args) == 2:
        # 如果有两个参数，第一个是名字，第二个是语言
        greet(args[0], language=args[1])
    else:
        print("用法: python -m my_toolkit.greeter <名字> [语言: en/es/zh]")