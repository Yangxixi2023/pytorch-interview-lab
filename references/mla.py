"""多头潜在注意力 MLA

输入：
q: 浮点 Tensor [B, H, T, D] — 已投影的 query。
latent: 浮点 Tensor [B, S, R] — 共享的低秩 KV 表示。
wk: 浮点 Tensor [H, R, D] — 每头 key 重构矩阵。
wv: 浮点 Tensor [H, R, D] — 每头 value 重构矩阵。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, H, T, D] — 重构 KV 后的非因果注意力输出，不含 RoPE。
"""

import math
import torch

def solve(q, latent, wk, wv):
    k = torch.einsum('bsr,hrd->bhsd', latent, wk)
    v = torch.einsum('bsr,hrd->bhsd', latent, wv)
    return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1) @ v

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, 5, requires_grad=True), torch.randn(2, 6, 2, requires_grad=True), torch.randn(3, 2, 5, requires_grad=True), torch.randn(3, 2, 5, requires_grad=True))
    print(solve(*args))
