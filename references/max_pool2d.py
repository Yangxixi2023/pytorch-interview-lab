"""二维最大池化

输入：
x: 浮点 Tensor [N, C, H, W] — 图像输入，无 padding。
kernel_size: int — — 正方形池化窗口边长。
stride: int — — 滑动步长。

返回（多项按元组顺序）：
output: 浮点 Tensor [N, C, Hout, Wout] — Hout=floor((H-kernel_size)/stride)+1，Wout 同理。
"""

import torch

def solve(x, kernel_size, stride):
    windows = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    return windows.flatten(-2).max(-1).values

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.arange(96.0).reshape(2, 3, 4, 4).requires_grad_(), 2, 2)
    print(solve(*args))
