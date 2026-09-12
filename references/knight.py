"""模拟马走日

输入：
rows: int —
    棋盘行数，正整数。
cols: int —
    棋盘列数，正整数。
start: tuple[int, int] (row, col)
    0-based 起点，位于棋盘内。
target: tuple[int, int] (row, col)
    0-based 终点，位于棋盘内。
blocked: list[tuple[int, int]] 任意长度
    不可落点坐标，允许为空。

返回：
min_steps: int —
    最少步数；不可达或起终点被阻挡返回 -1；合法同点返回0。
"""

import torch


def solve(rows, cols, start, target, blocked):
    from collections import deque

    start = tuple(start)
    target = tuple(target)
    blocked_cells = set(map(tuple, blocked))
    if start in blocked_cells or target in blocked_cells:
        return -1

    moves = [
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
    ]
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (row, column), distance = queue.popleft()
        if (row, column) == target:
            return distance

        for row_step, column_step in moves:
            next_row = row + row_step
            next_column = column + column_step
            inside_board = 0 <= next_row < rows and 0 <= next_column < cols
            next_cell = (next_row, next_column)
            if not inside_board:
                continue
            if next_cell in blocked_cells or next_cell in visited:
                continue
            visited.add(next_cell)  # 入队时标记，避免重复入队。
            queue.append((next_cell, distance + 1))

    return -1


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (8, 8, (0, 0), (7, 7), [])
    print(solve(*args))
