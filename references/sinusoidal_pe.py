"""正弦位置编码

输入：
length: int —
    序列长度。
dim: int —
    偶数特征维度。
base: float —
    频率底数。

返回：
encoding: 浮点 Tensor [length, dim]
    float32 位置编码，偶数列 sin、奇数列 cos。
"""

import torch


def solve(length, dim, base=10000.0):
    even_dimensions = torch.arange(0, dim, 2, dtype=torch.float32)
    inverse_frequencies = base ** (-even_dimensions / dim)
    positions = torch.arange(length)
    angles = positions[:, None] * inverse_frequencies[None, :]

    sine_features = torch.sin(angles)
    cosine_features = torch.cos(angles)
    # 每对按 sin、cos 交替排列，再展平成 [length,dim]。
    paired_features = torch.stack((sine_features, cosine_features), dim=-1)
    encoding = paired_features.flatten(start_dim=-2)
    return encoding


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (5, 8)
    print(solve(*args))
