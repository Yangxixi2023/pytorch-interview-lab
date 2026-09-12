"""差分注意力

输入：
q1: 浮点 Tensor [B, H, T, D]
    第一组 query。
k1: 浮点 Tensor [B, H, T, D]
    第一组 key。
q2: 浮点 Tensor [B, H, T, D]
    第二组 query。
k2: 浮点 Tensor [B, H, T, D]
    第二组 key。
v: 浮点 Tensor [B, H, T, D]
    两组注意力共享的 value。
lam: float —
    第二组 softmax 权重的减法系数。

返回：
output: 浮点 Tensor [B, H, T, D]
    两组注意力之差作用到 v 的输出；没有因果 mask 或 head norm。
"""

import math
import torch


def solve(q1, k1, q2, k2, v, lam):
    scale = math.sqrt(q1.shape[-1])
    first_scores = (q1 @ k1.transpose(-2, -1)) / scale
    second_scores = (q2 @ k2.transpose(-2, -1)) / scale
    first_weights = first_scores.softmax(dim=-1)
    second_weights = second_scores.softmax(dim=-1)

    differential_weights = first_weights - lam * second_weights
    output = differential_weights @ v
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple(
        (torch.randn(1, 2, 3, 4, requires_grad=True) for _ in range(5))
    ) + (0.8,)
    print(solve(*args))
