"""二维卷积

输入：
x: 浮点 Tensor [N, Cin, H, W]
    输入图像。
weight: 浮点 Tensor [Cout, Cin, Kh, Kw]
    互相关卷积核，不翻转。
bias: 浮点 Tensor [Cout]
    输出偏移。
stride: int —
    两个空间维度使用相同步长。
padding: int —
    四边相同的零 padding。

返回：
output: 浮点 Tensor [N, Cout, Hout, Wout]
    Hout=floor((H+2*padding-Kh)/stride)+1，Wout 同理；groups=1。
"""

import torch
import torch.nn.functional as F


def solve(x, weight, bias, stride=1, padding=0):
    kernel_height, kernel_width = weight.shape[-2:]
    padded_input = F.pad(x, (padding, padding, padding, padding))
    windows = padded_input.unfold(2, kernel_height, stride)
    windows = windows.unfold(3, kernel_width, stride)

    # windows=[N,Cin,Hout,Wout,Kh,Kw]，对 Cin、Kh、Kw 求和。
    output = torch.einsum("nchwij,ocij->nohw", windows, weight)
    output = output + bias[None, :, None, None]
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 2, 5, 6, requires_grad=True),
        torch.randn(3, 2, 2, 3, requires_grad=True),
        torch.randn(3, requires_grad=True),
        2,
        1,
    )
    print(solve(*args))
