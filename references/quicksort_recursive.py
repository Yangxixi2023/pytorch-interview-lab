"""快速排序（递归）

输入：
nums: list[int] 任意长度
    可为空，包含负数或重复元素；不修改输入。

返回：
sorted_nums: list[int] 与 nums 等长
    升序新列表，保留重复值。
"""

import torch


def solve(nums):
    def partition_sort(values):
        if len(values) <= 1:
            return values

        pivot = values[len(values) // 2]
        smaller = []
        equal = []
        larger = []
        for value in values:
            if value < pivot:
                smaller.append(value)
            elif value > pivot:
                larger.append(value)
            else:
                equal.append(value)

        return partition_sort(smaller) + equal + partition_sort(larger)

    # 先复制，避免修改调用者传入的列表。
    return partition_sort(list(nums))


if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([3, -1, 2, 3, 0, -1],)
    print(solve(*args))
