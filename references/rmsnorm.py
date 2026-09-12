"""实现 RMSNorm

输入：
x: 浮点 Tensor [..., D] — 沿最后一维计算均方，不减均值。
weight: 浮点 Tensor [D] — 逐特征缩放权重。
eps: float — — 加在均方内部的稳定项。

返回（多项按元组顺序）：
output: 浮点 Tensor 与 x 相同 — 归一化后乘 weight；x 和 weight 均可求导。
"""

import torch

def solve(x, weight, eps=1e-06):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, requires_grad=True), torch.randn(4, requires_grad=True))
    print(solve(*args))
