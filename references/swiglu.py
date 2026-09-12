"""SwiGLU 激活函数

输入：
x: 浮点 Tensor [..., 2D]
    最后一维为偶数；前半为 a，后半为 b。

返回：
output: 浮点 Tensor [..., D]
    SiLU(a)*b，最后一维减半，保留梯度。
"""

import torch


def solve(x):
    # 最后一维一分为二：[...,2D] -> 两个 [...,D]。
    gate_input, value_input = x.chunk(2, dim=-1)
    activated_gate = gate_input * torch.sigmoid(gate_input)
    output = activated_gate * value_input
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 6, requires_grad=True),)
    print(solve(*args))
