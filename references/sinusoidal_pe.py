"""正弦位置编码

输入：
length: int — — 序列长度。
dim: int — — 偶数特征维度。
base: float — — 频率底数。

返回（多项按元组顺序）：
encoding: 浮点 Tensor [length, dim] — float32 位置编码，偶数列 sin、奇数列 cos。
"""

import torch

def solve(length, dim, base=10000.0):
    freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    a = torch.arange(length)[:, None] * freq
    return torch.stack((a.sin(), a.cos()), -1).flatten(-2)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (5, 8)
    print(solve(*args))
