"""
N 皇后 - N-Queens
LeetCode 51

问题：在 n×n 棋盘上放 n 个皇后，使其互不攻击（不同行/列/对角线）。
"""


def solve_n_queens(n: int) -> list[list[str]]:
    result = []
    board = [["."] * n for _ in range(n)]
    cols, diag1, diag2 = set(), set(), set()

    def backtrack(row: int):
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            d1 = row - col
            d2 = row + col
            if col in cols or d1 in diag1 or d2 in diag2:
                continue
            # 做选择
            board[row][col] = "Q"
            cols.add(col)
            diag1.add(d1)
            diag2.add(d2)
            # 递归
            backtrack(row + 1)
            # 撤销选择
            board[row][col] = "."
            cols.remove(col)
            diag1.remove(d1)
            diag2.remove(d2)

    backtrack(0)
    return result


if __name__ == "__main__":
    n = 4
    solutions = solve_n_queens(n)
    print(f"{n} 皇后共有 {len(solutions)} 种解法:\n")
    for sol in solutions:
        for row in sol:
            print(row)
        print()
