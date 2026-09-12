"""GELU 激活函数

输入：
x: 浮点 Tensor 任意形状
    输入激活值，保留梯度。

返回：
output: 浮点 Tensor 与 x 相同
    逐元素激活，保持 dtype、device 和形状。
"""

import math
import torch


def solve(x):
    # 精确 GELU: x 乘标准正态分布的累积分布函数。
    normal_cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    output = x * normal_cdf
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 6, requires_grad=True),)
    print(solve(*args))
