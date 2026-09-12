"""因果自注意力

输入：
q: 浮点 Tensor [B, H, T, D]
    已拆头的 query。
k: 浮点 Tensor [B, H, T, D]
    已拆头的 key。
v: 浮点 Tensor [B, H, T, D]
    已拆头的 value。

返回：
output: 浮点 Tensor [B, H, T, D]
    因果注意力输出；不允许读取未来位置，保留梯度。
"""

import math
import torch


def solve(q, k, v):
    sequence_length = q.shape[-2]
    head_dim = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

    # 行是 query 位置，列是 key 位置；上三角表示未来。
    future_mask = torch.ones(
        sequence_length, sequence_length, device=q.device, dtype=torch.bool
    ).triu(diagonal=1)
    scores = scores.masked_fill(future_mask, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ v
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple(
        (torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3))
    )
    print(solve(*args))
