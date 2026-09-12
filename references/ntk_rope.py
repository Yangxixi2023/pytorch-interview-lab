"""NTK-aware RoPE 缩放

输入：
dim: int —
    偶数且大于 2。
scale: float —
    上下文缩放倍数，>=1。
base: float —
    原始 RoPE 底数。

返回：
inverse_frequencies: 浮点 Tensor [dim/2]
    float32 的缩放后逆频率。
"""

import torch


def solve(dim, scale, base=10000.0):
    # 本题是固定 NTK 缩放，不包含根据当前长度切换的动态策略。
    scaled_base = base * scale ** (dim / (dim - 2))
    even_dimensions = torch.arange(0, dim, 2, dtype=torch.float32)
    inverse_frequencies = scaled_base ** (-even_dimensions / dim)
    return inverse_frequencies


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (8, 4.0)
    print(solve(*args))
