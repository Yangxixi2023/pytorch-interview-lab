"""二维最大池化

输入：
x: 浮点 Tensor [N, C, H, W]
    图像输入，无 padding。
kernel_size: int —
    正方形池化窗口边长。
stride: int —
    滑动步长。

返回：
output: 浮点 Tensor [N, C, Hout, Wout]
    Hout=floor((H-kernel_size)/stride)+1，Wout 同理。
"""

import torch


def solve(x, kernel_size, stride):
    # 展开窗口后形状为 [N,C,Hout,Wout,Kh,Kw]。
    windows = x.unfold(dimension=2, size=kernel_size, step=stride)
    windows = windows.unfold(dimension=3, size=kernel_size, step=stride)
    flattened_windows = windows.flatten(start_dim=-2)
    # 只在每个窗口内部取最大值，不跨通道或相邻窗口求最大。
    output = flattened_windows.max(dim=-1).values
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.arange(96.0).reshape(2, 3, 4, 4).requires_grad_(), 2, 2)
    print(solve(*args))
