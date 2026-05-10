"""
反转字符串 - Reverse String

递归：s[-1] + reverse(s[:-1])
"""


def reverse_string(s: str) -> str:
    if len(s) <= 1:
        return s
    return s[-1] + reverse_string(s[:-1])


if __name__ == "__main__":
    test = "hello"
    print(f"reverse('{test}') = '{reverse_string(test)}'")
