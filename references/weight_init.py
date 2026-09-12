"""Kaiming 初始化

输入：
standard_normal: 浮点 Tensor 任意形状 — 已采样的标准正态噪声；不要再次采样。
fan_in: int — — 输入连接数，正整数。

返回（多项按元组顺序）：
weight: 浮点 Tensor 与 standard_normal 相同 — 按 sqrt(2/fan_in) 缩放。
"""

import math
import torch

def solve(standard_normal, fan_in):
    return standard_normal * math.sqrt(2 / fan_in)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(20, 30), 30)
    print(solve(*args))
