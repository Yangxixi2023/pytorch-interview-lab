"""多头注意力

输入：
x: 浮点 Tensor [B, T, D] — 未投影输入。
wq: 浮点 Tensor [D, D] — 按 x @ wq 使用；不转置。
wk: 浮点 Tensor [D, D] — 按 x @ wk 使用；不转置。
wv: 浮点 Tensor [D, D] — 按 x @ wv 使用；不转置。
wo: 浮点 Tensor [D, D] — 按 x @ wo 使用；不转置。
heads: int — — head 数 H，D 必须可被 H 整除。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, T, D] — 非因果多头注意力合并后经过 wo 投影。
"""

import math
import torch

def solve(x, wq, wk, wv, wo, heads):
    b, t, d = x.shape
    q, k, v = [(x @ w).reshape(b, t, heads, d // heads).transpose(1, 2) for w in (wq, wk, wv)]
    y = (q @ k.transpose(-2, -1) / math.sqrt(d // heads)).softmax(-1) @ v
    return y.transpose(1, 2).reshape(b, t, d) @ wo

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 8, requires_grad=True),) + tuple((torch.randn(8, 8, requires_grad=True) * 0.2 for _ in range(4))) + (2,)
    print(solve(*args))
