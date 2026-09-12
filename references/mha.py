"""多头注意力

输入：
x: 浮点 Tensor [B, T, D]
    未投影输入。
wq: 浮点 Tensor [D, D]
    按 x @ wq 使用；不转置。
wk: 浮点 Tensor [D, D]
    按 x @ wk 使用；不转置。
wv: 浮点 Tensor [D, D]
    按 x @ wv 使用；不转置。
wo: 浮点 Tensor [D, D]
    按 x @ wo 使用；不转置。
heads: int —
    head 数 H，D 必须可被 H 整除。

返回：
output: 浮点 Tensor [B, T, D]
    非因果多头注意力合并后经过 wo 投影。
"""

import math
import torch


def solve(x, wq, wk, wv, wo, heads):
    batch_size, sequence_length, hidden_dim = x.shape
    head_dim = hidden_dim // heads

    # 先独立投影，再拆头：[B,T,D] -> [B,H,T,Dh]。
    query = x @ wq
    key = x @ wk
    value = x @ wv
    query = query.reshape(batch_size, sequence_length, heads, head_dim)
    key = key.reshape(batch_size, sequence_length, heads, head_dim)
    value = value.reshape(batch_size, sequence_length, heads, head_dim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
    attention_weights = torch.softmax(scores, dim=-1)
    head_outputs = attention_weights @ value

    # 合并前要先交换回时间维；reshape 会处理非连续内存。
    merged = head_outputs.transpose(1, 2)
    merged = merged.reshape(batch_size, sequence_length, hidden_dim)
    output = merged @ wo
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        (torch.randn(2, 3, 8, requires_grad=True),)
        + tuple(
            (torch.randn(8, 8, requires_grad=True) * 0.2 for _ in range(4))
        )
        + (2,)
    )
    print(solve(*args))
