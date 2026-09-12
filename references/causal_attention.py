"""因果自注意力

输入：
q: 浮点 Tensor [B, H, T, D] — 已拆头的 query。
k: 浮点 Tensor [B, H, T, D] — 已拆头的 key。
v: 浮点 Tensor [B, H, T, D] — 已拆头的 value。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, H, T, D] — 因果注意力输出；不允许读取未来位置，保留梯度。
"""

import math
import torch

def solve(q, k, v):
    t = q.shape[-2]
    i = torch.arange(t, device=q.device)[:, None]
    j = torch.arange(t, device=q.device)[None, :]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return scores.masked_fill(j > i, float('-inf')).softmax(-1) @ v

if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple((torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3)))
    print(solve(*args))
