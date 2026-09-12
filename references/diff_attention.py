"""差分注意力

输入：
q1: 浮点 Tensor [B, H, T, D] — 第一组 query。
k1: 浮点 Tensor [B, H, T, D] — 第一组 key。
q2: 浮点 Tensor [B, H, T, D] — 第二组 query。
k2: 浮点 Tensor [B, H, T, D] — 第二组 key。
v: 浮点 Tensor [B, H, T, D] — 两组注意力共享的 value。
lam: float — — 第二组 softmax 权重的减法系数。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, H, T, D] — 两组注意力之差作用到 v 的输出；没有因果 mask 或 head norm。
"""

import math
import torch

def solve(q1, k1, q2, k2, v, lam):
    s = math.sqrt(q1.shape[-1])
    return ((q1 @ k1.transpose(-2, -1) / s).softmax(-1) - lam * (q2 @ k2.transpose(-2, -1) / s).softmax(-1)) @ v

if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple((torch.randn(1, 2, 3, 4, requires_grad=True) for _ in range(5))) + (0.8,)
    print(solve(*args))
