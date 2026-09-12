"""旋转位置编码 RoPE

输入：
x: 浮点 Tensor [B, H, T, D] — D 为偶数，采用相邻偶奇维配对。
positions: 整数 Tensor [T] — 各 token 的绝对位置，可从非零位置开始。
base: float — — 旋转频率的底数。

返回（多项按元组顺序）：
output: 浮点 Tensor 与 x 相同 — 旋转后的张量；保留 x 的梯度。
"""

import torch

def solve(x, positions, base=10000.0):
    d = x.shape[-1]
    freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angle = positions[:, None] * freq
    c = angle.cos()
    s = angle.sin()
    a = x[..., 0::2]
    b = x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 2, 3, 8, requires_grad=True), torch.tensor([3, 4, 5]))
    print(solve(*args))
