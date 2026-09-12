"""FFN 前馈网络

输入：
x: 浮点 Tensor [..., D] — 输入。
w1: 浮点 Tensor [D, F] — 上投影。
b1: 浮点 Tensor [F] — 上投影偏移。
w2: 浮点 Tensor [F, D] — 下投影。
b2: 浮点 Tensor [D] — 输出偏移。

返回（多项按元组顺序）：
output: 浮点 Tensor 与 x 相同 — 精确 GELU 的双线性层输出，保留梯度，不包含残差。
"""

import math
import torch

def solve(x, w1, b1, w2, b2):
    z = x @ w1 + b1
    g = 0.5 * z * (1 + torch.erf(z / math.sqrt(2)))
    return g @ w2 + b2

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, requires_grad=True), torch.randn(4, 7, requires_grad=True), torch.randn(7, requires_grad=True), torch.randn(7, 4, requires_grad=True), torch.randn(4, requires_grad=True))
    print(solve(*args))
