"""FSDP 训练步骤（单机模拟）

输入：
param_shards: list[Tensor] 第 i 项 [Si]
    一维参数分片，完整长度 S=sum(Si)。
rank_grads: list[Tensor] 每项 [S]
    每个 rank 对完整参数的梯度。
lr: float —
    SGD 学习率。

返回：
updated_shards: list[Tensor] 与 param_shards 一一对应
    先跨 rank 平均完整梯度，再按原长度切片更新；不修改输入。
"""

import torch


def solve(param_shards, rank_grads, lr):
    mean_gradient = torch.stack(rank_grads, dim=0).mean(dim=0)
    updated_shards = []
    offset = 0

    for parameter_shard in param_shards:
        shard_size = parameter_shard.numel()
        gradient_shard = mean_gradient[offset : offset + shard_size]
        updated_shard = parameter_shard - lr * gradient_shard
        updated_shards.append(updated_shard)
        offset += shard_size

    return updated_shards


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        [torch.ones(2), torch.ones(3)],
        [torch.arange(5.0), torch.arange(5.0) * 3],
        0.1,
    )
    print(solve(*args))
