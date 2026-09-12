"""ViT Transformer Block

输入：
x: 浮点 Tensor [B, T, D] — block 输入。
wqkv: 浮点 Tensor [D, 3D] — 联合 QKV 投影，按最后一维分成 q/k/v。
wo: 浮点 Tensor [D, D] — 注意力输出投影。
w1: 浮点 Tensor [D, F] — MLP 上投影。
w2: 浮点 Tensor [F, D] — MLP 下投影。
heads: int — — head 数，必须整除 D。
eps: float — — 两处无仿射 LayerNorm 的稳定项。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, T, D] — pre-LN 双残差 block 输出；GPT-2 因果，ViT 双向。
"""

import math
import torch
import torch.nn.functional as F

def solve(x, wqkv, wo, w1, w2, heads, eps=1e-05):
    b, t, d = x.shape
    z = F.layer_norm(x, (d,), eps=eps)
    q, k, v = [part.reshape(b, t, heads, d // heads).transpose(1, 2) for part in (z @ wqkv).chunk(3, -1)]
    s = q @ k.transpose(-2, -1) / math.sqrt(d // heads)
    a = x + (s.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d) @ wo
    return a + F.gelu(F.layer_norm(a, (d,), eps=eps) @ w1) @ w2

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, requires_grad=True), torch.randn(4, 12, requires_grad=True) * 0.2, torch.randn(4, 4, requires_grad=True) * 0.2, torch.randn(4, 6, requires_grad=True) * 0.2, torch.randn(6, 4, requires_grad=True) * 0.2, 2)
    print(solve(*args))
