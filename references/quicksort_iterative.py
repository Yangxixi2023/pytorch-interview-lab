"""快速排序（非递归）

输入：
nums: list[int] 任意长度 — 可为空，包含负数或重复元素；不修改输入。

返回（多项按元组顺序）：
sorted_nums: list[int] 与 nums 等长 — 升序新列表，保留重复值。
"""

import torch

def solve(nums):
    a = list(nums)
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            if a[j] < pivot:
                a[i], a[j] = (a[j], a[i])
                i += 1
        a[i], a[hi] = (a[hi], a[i])
        stack.extend([(lo, i - 1), (i + 1, hi)])
    return a

if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([3, -1, 2, 3, 0, -1],)
    print(solve(*args))
