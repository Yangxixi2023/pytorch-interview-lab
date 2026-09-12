"""环形注意力（单机模拟）

输入：
q: 浮点 Tensor [Q, D]
    单头 query，无 batch 维。
k: 浮点 Tensor [K, D]
    按序列维分块处理的 key。
v: 浮点 Tensor [K, V]
    按同样块划分的 value。
block_size: int —
    正的 KV 分块大小，K 不一定可整除。

返回：
output: 浮点 Tensor [Q, V]
    精确非因果注意力，不构造完整 Q×K 分数矩阵。
"""

import math
import torch


def solve(q, k, v, block_size=2):
    query_count = q.shape[0]
    value_dim = v.shape[-1]
    running_max = q.new_full((query_count, 1), float("-inf"))
    running_sum = q.new_zeros(query_count, 1)
    running_values = q.new_zeros(query_count, value_dim)

    for start in range(0, k.shape[0], block_size):
        key_block = k[start : start + block_size]
        value_block = v[start : start + block_size]
        block_scores = (q @ key_block.transpose(0, 1)) / math.sqrt(
            q.shape[-1]
        )

        block_max = block_scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        # 新最大值改变后，把旧累计量换算到相同的指数尺度。
        old_scale = torch.exp(running_max - new_max)
        block_exponentials = torch.exp(block_scores - new_max)
        running_values = running_values * old_scale
        running_values = running_values + block_exponentials @ value_block
        running_sum = running_sum * old_scale
        running_sum = running_sum + block_exponentials.sum(
            dim=-1, keepdim=True
        )
        running_max = new_max

    output = running_values / running_sum
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(4, 3, requires_grad=True),
        torch.randn(7, 3, requires_grad=True),
        torch.randn(7, 5, requires_grad=True),
        3,
    )
    print(solve(*args))
