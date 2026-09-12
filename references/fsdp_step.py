"""FSDP 训练步骤（单机模拟）

输入：
param_shards: list[Tensor] 第 i 项 [Si] — 一维参数分片，完整长度 S=sum(Si)。
rank_grads: list[Tensor] 每项 [S] — 每个 rank 对完整参数的梯度。
lr: float — — SGD 学习率。

返回（多项按元组顺序）：
updated_shards: list[Tensor] 与 param_shards 一一对应 — 先跨 rank 平均完整梯度，再按原长度切片更新；不修改输入。
"""

import torch

def solve(param_shards, rank_grads, lr):
    g = torch.stack(rank_grads).mean(0)
    out = []
    start = 0
    for p in param_shards:
        out.append(p - lr * g[start:start + p.numel()])
        start += p.numel()
    return out

if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([torch.ones(2), torch.ones(3)], [torch.arange(5.0), torch.arange(5.0) * 3], 0.1)
    print(solve(*args))
