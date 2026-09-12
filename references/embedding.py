"""Embedding 层

输入：
ids: int64 Tensor 任意形状 — 词表索引，取值 0..V-1；可以重复。
weight: 浮点 Tensor [V, D] — 可训练的 embedding 表。

返回（多项按元组顺序）：
output: 浮点 Tensor [*ids.shape, D] — 查表结果；重复索引对应 weight 梯度需累加。
"""

import torch

def solve(ids, weight):
    return weight[ids]

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.tensor([[1, 1, 3], [0, 2, 1]]), torch.randn(5, 4, requires_grad=True))
    print(solve(*args))
