"""Top-k 最大元素

输入：
nums: list[int] 任意长度 — 允许负数和重复元素。
k: int — — 0<=k<=len(nums)。

返回（多项按元组顺序）：
largest: list[int] k 项 — 最大的 k 个元素，降序排列，保留重复。
"""

import torch

def solve(nums, k):
    import heapq
    if k == 0:
        return []
    heap = []
    for n in nums:
        if len(heap) < k:
            heapq.heappush(heap, n)
        elif n > heap[0]:
            heapq.heapreplace(heap, n)
    return [heapq.heappop(heap) for _ in range(k)][::-1]

if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([2, 7, 3, 7, -1], 3)
    print(solve(*args))
