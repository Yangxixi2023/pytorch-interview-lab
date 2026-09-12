"""模拟马走日

输入：
rows: int — — 棋盘行数，正整数。
cols: int — — 棋盘列数，正整数。
start: tuple[int, int] (row, col) — 0-based 起点，位于棋盘内。
target: tuple[int, int] (row, col) — 0-based 终点，位于棋盘内。
blocked: list[tuple[int, int]] 任意长度 — 不可落点坐标，允许为空。

返回（多项按元组顺序）：
min_steps: int — — 最少步数；不可达或起终点被阻挡返回 -1；合法同点返回0。
"""

import torch

def solve(rows, cols, start, target, blocked):
    from collections import deque
    blocked = set(map(tuple, blocked))
    start = tuple(start)
    target = tuple(target)
    if start in blocked or target in blocked:
        return -1
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == target:
            return dist
        for dr, dc in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and (nxt not in seen) and (nxt not in blocked):
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return -1

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (8, 8, (0, 0), (7, 7), [])
    print(solve(*args))
