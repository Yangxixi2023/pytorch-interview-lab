"""快速排序（非递归）

输入：
nums: list[int] 任意长度
    可为空，包含负数或重复元素；不修改输入。

返回：
sorted_nums: list[int] 与 nums 等长
    升序新列表，保留重复值。
"""

import torch


def solve(nums):
    values = list(nums)
    pending_ranges = [(0, len(values) - 1)]

    while pending_ranges:
        left, right = pending_ranges.pop()
        if left >= right:
            continue

        pivot = values[right]
        boundary = left
        # [left,boundary) 始终存放小于 pivot 的元素。
        for scan in range(left, right):
            if values[scan] < pivot:
                values[boundary], values[scan] = (
                    values[scan],
                    values[boundary],
                )
                boundary += 1
        values[boundary], values[right] = values[right], values[boundary]

        pending_ranges.append((left, boundary - 1))
        pending_ranges.append((boundary + 1, right))

    return values


if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([3, -1, 2, 3, 0, -1],)
    print(solve(*args))
