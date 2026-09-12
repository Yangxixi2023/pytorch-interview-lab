"""二维卷积

输入：
x: 浮点 Tensor [N, Cin, H, W] — 输入图像。
weight: 浮点 Tensor [Cout, Cin, Kh, Kw] — 互相关卷积核，不翻转。
bias: 浮点 Tensor [Cout] — 输出偏移。
stride: int — — 两个空间维度使用相同步长。
padding: int — — 四边相同的零 padding。

返回（多项按元组顺序）：
output: 浮点 Tensor [N, Cout, Hout, Wout] — Hout=floor((H+2*padding-Kh)/stride)+1，Wout 同理；groups=1。
"""

import torch
import torch.nn.functional as F

def solve(x, weight, bias, stride=1, padding=0):
    kh, kw = weight.shape[-2:]
    z = F.pad(x, (padding, padding, padding, padding))
    windows = z.unfold(2, kh, stride).unfold(3, kw, stride)
    return torch.einsum('nchwij,ocij->nohw', windows, weight) + bias[None, :, None, None]

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 2, 5, 6, requires_grad=True), torch.randn(3, 2, 2, 3, requires_grad=True), torch.randn(3, requires_grad=True), 2, 1)
    print(solve(*args))
