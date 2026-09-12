"""简单线性层

输入：
x: 浮点 Tensor [..., Din] — 任意前导 batch 维度。
weight: 浮点 Tensor [Dout, Din] — 注意权重以输出维度在前。
bias: 浮点 Tensor [Dout] — 输出偏移。

返回（多项按元组顺序）：
output: 浮点 Tensor [..., Dout] — x @ weight.T + bias，保留梯度。
"""

import torch

def solve(x, weight, bias):
    return x @ weight.T + bias

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, requires_grad=True), torch.randn(5, 4, requires_grad=True), torch.randn(5, requires_grad=True))
    print(solve(*args))
