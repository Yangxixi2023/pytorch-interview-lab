"""SwiGLU 激活函数

输入：
x: 浮点 Tensor [..., 2D] — 最后一维为偶数；前半为 a，后半为 b。

返回（多项按元组顺序）：
output: 浮点 Tensor [..., D] — SiLU(a)*b，最后一维减半，保留梯度。
"""

import torch

def solve(x):
    a, b = x.chunk(2, dim=-1)
    return a * torch.sigmoid(a) * b

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 6, requires_grad=True),)
    print(solve(*args))
