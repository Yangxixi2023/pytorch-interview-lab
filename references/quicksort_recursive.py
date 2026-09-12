"""快速排序（递归）

输入：
nums: list[int] 任意长度 — 可为空，包含负数或重复元素；不修改输入。

返回（多项按元组顺序）：
sorted_nums: list[int] 与 nums 等长 — 升序新列表，保留重复值。
"""

import torch

def solve(nums):

    def sort(a):
        if len(a) < 2:
            return a
        p = a[len(a) // 2]
        return sort([v for v in a if v < p]) + [v for v in a if v == p] + sort([v for v in a if v > p])
    return sort(list(nums))

if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([3, -1, 2, 3, 0, -1],)
    print(solve(*args))
