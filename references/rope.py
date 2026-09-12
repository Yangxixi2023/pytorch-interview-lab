"""旋转位置编码 RoPE

输入：
x: 浮点 Tensor [B, H, T, D]
    D 为偶数，采用相邻偶奇维配对。
positions: 整数 Tensor [T]
    各 token 的绝对位置，可从非零位置开始。
base: float —
    旋转频率的底数。

返回：
output: 浮点 Tensor 与 x 相同
    旋转后的张量；保留 x 的梯度。
"""

import torch


def solve(x, positions, base=10000.0):
    hidden_dim = x.shape[-1]
    even_dimensions = torch.arange(
        0, hidden_dim, 2, device=x.device, dtype=x.dtype
    )
    inverse_frequencies = base ** (-even_dimensions / hidden_dim)
    angles = positions[:, None] * inverse_frequencies  # [T,D/2]
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    # 本题采用相邻偶奇配对，而不是前后半维配对。
    even_values = x[..., 0::2]
    odd_values = x[..., 1::2]
    rotated_even = even_values * cosine - odd_values * sine
    rotated_odd = even_values * sine + odd_values * cosine

    paired_output = torch.stack((rotated_even, rotated_odd), dim=-1)
    output = paired_output.flatten(start_dim=-2)
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 2, 3, 8, requires_grad=True),
        torch.tensor([3, 4, 5]),
    )
    print(solve(*args))
