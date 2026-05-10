"""
括号生成 - Generate Parentheses
LeetCode 22

问题：n 对括号，生成所有有效括号组合。
"""


def generate_parenthesis(n: int) -> list[str]:
    result = []

    def backtrack(curr: str, left: int, right: int):
        if len(curr) == 2 * n:
            result.append(curr)
            return
        if left < n:
            backtrack(curr + "(", left + 1, right)
        if right < left:
            backtrack(curr + ")", left, right + 1)

    backtrack("", 0, 0)
    return result


if __name__ == "__main__":
    n = 3
    result = generate_parenthesis(n)
    print(f"n = {n}: {result}")
    # 预期: ['((()))', '(()())', '(())()', '()(())', '()()()']
