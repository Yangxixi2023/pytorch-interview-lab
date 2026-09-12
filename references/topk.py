"""Top-k 最大元素

输入：
nums: list[int] 任意长度
    允许负数和重复元素。
k: int —
    0<=k<=len(nums)。

返回：
largest: list[int] k 项
    最大的 k 个元素，降序排列，保留重复。
"""

import torch


def solve(nums, k):
    import heapq

    if k == 0:
        return []

    largest_values = []
    for value in nums:
        if len(largest_values) < k:
            heapq.heappush(largest_values, value)
        elif value > largest_values[0]:
            # 堆顶是当前前 k 大元素中最小的一个。
            heapq.heapreplace(largest_values, value)

    ascending_result = []
    while largest_values:
        ascending_result.append(heapq.heappop(largest_values))
    return ascending_result[::-1]


if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([2, 7, 3, 7, -1], 3)
    print(solve(*args))
