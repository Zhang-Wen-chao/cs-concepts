"""
回文判断 - Palindrome

递归：首尾相等 且 中间也是回文
"""


def is_palindrome(s: str) -> bool:
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    return is_palindrome(s[1:-1])


if __name__ == "__main__":
    for test in ["racecar", "hello", "a", ""]:
        result = is_palindrome(test)
        print(f"is_palindrome('{test}') = {result}")
