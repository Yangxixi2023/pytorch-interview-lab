"""ALiBi 注意力

输入：
q: 浮点 Tensor [B, H, T, D] — 已拆头的 query。
k: 浮点 Tensor [B, H, T, D] — 已拆头的 key。
v: 浮点 Tensor [B, H, T, D] — 已拆头的 value。
slopes: 浮点 Tensor [H] — 每个 head 的位置惩罚斜率。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, H, T, D] — 因果注意力输出；不允许读取未来位置，保留梯度。
"""

import math
import torch

def solve(q, k, v, slopes):
    t = q.shape[-2]
    i = torch.arange(t, device=q.device)[:, None]
    j = torch.arange(t, device=q.device)[None, :]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores - slopes[None, :, None, None] * (i - j)
    return scores.masked_fill(j > i, float('-inf')).softmax(-1) @ v

if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple((torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3))) + (torch.tensor([0.1, 0.3]),)
    print(solve(*args))
